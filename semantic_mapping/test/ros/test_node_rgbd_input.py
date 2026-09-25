"""Sensor input variants: best-effort QoS by default, and CompressedImage RGB
with a colour-aligned 16-bit depth image under reliable QoS."""
import time

import numpy as np
import pytest
import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Header

from semantic_mapping.ros_msgs import numpy_to_image
from test.ros.helpers import camera_info, set_camera_tf, spin_until, stamp_msg

cv2 = pytest.importorskip("cv2")


def _png(rgb: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
    assert ok
    return encoded.tobytes()


def test_sensor_subscriptions_default_to_best_effort(node_factory):
    node = node_factory()
    for subscriber in node._sensor_subscribers:
        infos = node.get_subscriptions_info_by_topic(subscriber.topic)
        assert infos and infos[0].qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT, subscriber.topic


def test_compressed_rgb_with_aligned_depth_image(node_factory, dataset):
    node = node_factory("-p", "rgb_compressed:=true", "-p", "depth_source:=depth_image",
                        "-p", "depth_topic:=/camera/depth", "-p", "sensor_qos:=reliable",
                        silence_publishers=False)
    infos = node.get_subscriptions_info_by_topic("/camera/depth")
    assert infos and infos[0].qos_profile.reliability == ReliabilityPolicy.RELIABLE
    assert infos[0].topic_type == "sensor_msgs/msg/Image"
    assert node.get_subscriptions_info_by_topic("/camera/color/image_raw")[0].topic_type == "sensor_msgs/msg/CompressedImage"

    for publisher in (node.obj_boxes_pub, node.obj_points_pub):
        publisher.publish = lambda message: None
    annotated = []
    node.annotated_image_pub.publish = annotated.append
    node.annotated_image_pub.get_subscription_count = lambda: 1

    frames = list(dataset)[:5]
    intr = dataset.intrinsics
    for frame in frames:
        set_camera_tf(node, frame)
        header = Header(stamp=stamp_msg(frame.stamp), frame_id=node.camera_frame)
        rgb_msg = CompressedImage(header=header, format="rgb8; png compressed bgr8", data=_png(frame.rgb))
        depth_msg = numpy_to_image(np.round(frame.depth * 1000.0).astype(np.uint16), "16UC1", header)
        odom = Odometry(header=Header(stamp=stamp_msg(frame.stamp), frame_id="map"), child_frame_id="sensor")
        node._on_synced_frame(rgb_msg, camera_info(intr, header), depth_msg, odom)
        time.sleep(0.15)
    spin_until(node, lambda: not node._pending_frames)

    objects = node.pipeline.object_map.objects.values()
    assert len(objects) >= 5 and all(o.status.value == "active" for o in objects)
    assert annotated and annotated[-1].encoding == "rgb8" and annotated[-1].width == intr.width

    decoded = node._decode_rgb(CompressedImage(format="png", data=_png(frames[0].rgb)), intr)
    assert np.array_equal(decoded, frames[0].rgb)  # PNG is lossless


@pytest.mark.parametrize("sync_odometry", [False, True])
def test_rgbd_transport_without_odometry(node_factory, dataset, sync_odometry):
    """Exercise real DDS and synchronization, including the default odometry gate."""
    node = node_factory("-p", "depth_source:=depth_image",
                        "-p", f"sync_odometry:={str(sync_odometry).lower()}")
    frame = next(iter(dataset))
    set_camera_tf(node, frame)
    header = Header(stamp=stamp_msg(frame.stamp), frame_id=node.camera_frame)
    messages = [
        numpy_to_image(frame.rgb, "rgb8", header),
        camera_info(dataset.intrinsics, header),
        numpy_to_image(frame.depth.astype(np.float32), "32FC1", header),
    ]
    publishers = [node.create_publisher(msg_type, node.get_parameter(param).value, qos_profile_sensor_data)
                  for msg_type, param in [(Image, "rgb_topic"), (CameraInfo, "camera_info_topic"),
                                          (Image, "depth_topic")]]
    spin_until(node, lambda: all(pub.get_subscription_count() for pub in publishers))
    for pub, msg in zip(publishers, messages):
        pub.publish(msg)
    if sync_odometry:
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
        assert node._last_result is None
        odom_pub = node.create_publisher(Odometry, node.get_parameter("odometry_topic").value,
                                         qos_profile_sensor_data)
        spin_until(node, lambda: odom_pub.get_subscription_count() > 0)
        odom_pub.publish(Odometry(header=header))
    spin_until(node, lambda: node._last_result is not None)
    assert node._last_result.stamp == pytest.approx(frame.stamp)
    assert len(node._last_result.objects) >= 5


def test_disabling_odometry_sync_still_requires_camera_tf(node_factory, dataset):
    node = node_factory("-p", "depth_source:=depth_image", "-p", "sync_odometry:=false")
    frame = next(iter(dataset))
    header = Header(stamp=stamp_msg(frame.stamp), frame_id=node.camera_frame)
    node._on_synced_frame(
        numpy_to_image(frame.rgb, "rgb8", header), camera_info(dataset.intrinsics, header),
        numpy_to_image(frame.depth.astype(np.float32), "32FC1", header))
    assert node._last_result is None
    assert not node._pending_frames


@pytest.mark.parametrize('depth_source', ['depth_image', 'pointcloud'])
@pytest.mark.parametrize('binning', [1, 2])
def test_cropped_camera_input_preserves_pixels_and_mapped_geometry(node_factory, depth_source, binning):
    from types import SimpleNamespace
    from sensor_msgs.msg import RegionOfInterest
    from sensor_msgs_py import point_cloud2 as pc2
    from semantic_mapping.types import Detection2D

    node = node_factory('-p', f'depth_source:={depth_source}', '-p', 'min_hits_to_confirm:=1',
                        '-p', 'voxel_size:=0.001')
    T = np.eye(4)
    T[:3, 3] = [1., 2., 3.]
    node._lookup_se3 = lambda *args: T
    header = Header(stamp=stamp_msg(10.), frame_id=node.camera_frame)
    info = CameraInfo(header=header, width=256, height=128, binning_x=binning, binning_y=binning,
                      k=[128., 0., 127.5, 0., 96., 63.5, 0., 0., 1.],
                      roi=RegionOfInterest(x_offset=32, y_offset=8, width=192, height=96))
    h, w = 96 // binning, 192 // binning
    rgb = (np.arange(h * w * 3) % 256).astype(np.uint8).reshape(h, w, 3)
    mask = np.zeros((h, w), dtype=bool)
    mask[12:20, 28:40] = True
    depth = np.where(mask, 2., 0.).astype(np.float32)
    ys, xs = np.nonzero(mask)
    points = np.column_stack(((32 + xs * binning - 127.5) * 2 / 128,
                              (8 + ys * binning - 63.5) * 2 / 96, np.full(len(xs), 2.)))
    detection = Detection2D(np.array([28., 12., 40., 20.]), 'person', .99, mask=mask)
    node.detector = SimpleNamespace(detect=lambda *args, **kwargs: [detection])
    observations = []
    process = node._process_and_publish

    def record(obs, hdr):
        observations.append(obs)
        return process(obs, hdr)

    node._process_and_publish = record
    # Cover both raw and compressed images without hiding a resize in the decoder.
    rgb_msg = (CompressedImage(header=header, format='png', data=_png(rgb)) if binning == 2
               else numpy_to_image(rgb, 'rgb8', header))
    depth_msg = (numpy_to_image(depth, '32FC1', header) if depth_source == 'depth_image'
                 else pc2.create_cloud_xyz32(header, points))
    node._on_synced_frame(rgb_msg, info, depth_msg)
    spin_until(node, lambda: node._last_result is not None)
    assert len(observations) == 1
    np.testing.assert_array_equal(observations[0].rgb, rgb)
    np.testing.assert_allclose(observations[0].depth, depth)
    obj, = node._last_result.objects
    expected = points + T[:3, 3]
    ordered = lambda p: p[np.lexsort((p[:, 1], p[:, 0]))]
    np.testing.assert_allclose(ordered(obj.points_world), ordered(expected), atol=1e-6)


@pytest.mark.parametrize('invalid', ['roi', 'calibration', 'image_size'])
def test_bad_camera_input_is_skipped_and_next_valid_frame_recovers(node_factory, dataset, invalid):
    node = node_factory('-p', 'depth_source:=depth_image')
    frame = next(iter(dataset))
    set_camera_tf(node, frame)
    header = Header(stamp=stamp_msg(frame.stamp), frame_id=node.camera_frame)
    info = camera_info(dataset.intrinsics, header)
    if invalid == 'roi':
        info.roi.width = info.width + 1
        info.roi.height = info.height
    elif invalid == 'calibration':
        info.k[0] = 0.
    rgb = frame.rgb[:-1] if invalid == 'image_size' else frame.rgb
    depth_msg = numpy_to_image(frame.depth.astype(np.float32), '32FC1', header)
    node._on_synced_frame(numpy_to_image(rgb, 'rgb8', header), info, depth_msg)
    assert node._last_result is None and not node._pending_frames and node._next_frame_id == 0
    node._on_synced_frame(numpy_to_image(frame.rgb, 'rgb8', header), camera_info(dataset.intrinsics, header), depth_msg)
    spin_until(node, lambda: node._last_result is not None)
    assert node.pipeline._frame_index == 1
