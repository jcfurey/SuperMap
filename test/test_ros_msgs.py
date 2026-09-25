"""Message conversions; runs where ROS 2 message packages are importable (e.g. the Docker image)."""
import numpy as np
import pytest

sensor_msgs = pytest.importorskip("sensor_msgs.msg")

from semantic_mapping import ros_msgs  # noqa: E402


def test_image_round_trip_rgb8_and_bgr8_and_padding():
    rgb = (np.arange(2 * 3 * 3) % 255).astype(np.uint8).reshape(2, 3, 3)
    msg = ros_msgs.numpy_to_image(rgb, "rgb8")
    assert np.array_equal(ros_msgs.image_to_numpy(msg), rgb)

    bgr = ros_msgs.numpy_to_image(rgb[:, :, ::-1].copy(), "bgr8")
    assert np.array_equal(ros_msgs.image_to_numpy(bgr), rgb)

    padded = ros_msgs.numpy_to_image(rgb, "rgb8")
    padded.step = 3 * 3 + 4  # 4 bytes of row padding
    rows = np.frombuffer(padded.data, dtype=np.uint8).reshape(2, 9)
    padded.data = np.concatenate([rows, np.zeros((2, 4), np.uint8)], axis=1).tobytes()
    assert np.array_equal(ros_msgs.image_to_numpy(padded), rgb)


def test_depth_image_scaling_and_invalid_values():
    depth_mm = np.array([[0, 1500], [65535, 250]], dtype=np.uint16)
    msg = ros_msgs.numpy_to_image(depth_mm, "16UC1")
    depth = ros_msgs.depth_image_to_meters(msg, depth_scale=1000.0)
    assert depth.dtype == np.float32 and np.allclose(depth, [[0.0, 1.5], [65.535, 0.25]])

    depth_f = np.array([[np.nan, 2.0]], dtype=np.float32)
    msg = ros_msgs.numpy_to_image(depth_f, "32FC1")
    assert np.allclose(ros_msgs.depth_image_to_meters(msg), [[0.0, 2.0]])


def test_bayer_rggb8_is_demosaiced_to_rgb():
    mosaic = np.empty((8, 8), dtype=np.uint8)
    mosaic[0::2, 0::2] = 240  # R
    mosaic[0::2, 1::2] = 120  # G
    mosaic[1::2, 0::2] = 120  # G
    mosaic[1::2, 1::2] = 20   # B
    msg = ros_msgs.numpy_to_image(mosaic, "bayer_rggb8")
    rgb = ros_msgs.image_to_numpy(msg)
    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.uint8
    assert np.array_equal(rgb[3, 3], [240, 120, 20])


def test_camera_info_and_transform_conversions():
    info = sensor_msgs.CameraInfo(width=4, height=3, k=[10.0, 0.0, 2.0, 0.0, 12.0, 1.5, 0.0, 0.0, 1.0])
    intr = ros_msgs.camera_info_to_intrinsics(info)
    assert (intr.fx, intr.fy, intr.cx, intr.cy, intr.width, intr.height) == (10.0, 12.0, 2.0, 1.5, 4, 3)

    from geometry_msgs.msg import TransformStamped

    tf = TransformStamped()
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = 1.0, 2.0, 3.0
    tf.transform.rotation.w = 1.0
    T = ros_msgs.transform_to_se3(tf)
    assert np.allclose(T[:3, 3], [1.0, 2.0, 3.0]) and np.allclose(T[:3, :3], np.eye(3))


def test_pointcloud_to_xyz_drops_nans():
    from sensor_msgs_py import point_cloud2 as pc2
    from std_msgs.msg import Header

    cloud = pc2.create_cloud_xyz32(Header(frame_id="x"), [[1.0, 2.0, 3.0], [float("nan"), 0.0, 0.0], [4.0, 5.0, 6.0]])
    xyz = ros_msgs.pointcloud_to_xyz(cloud)
    assert xyz.shape == (2, 3) and np.allclose(xyz, [[1, 2, 3], [4, 5, 6]])


@pytest.mark.parametrize('binning', [(0, 0), (2, 3)])
def test_cropped_camera_info_preserves_full_sensor_rays(binning):
    import copy
    from semantic_mapping.geometry_utils import back_project_depth

    info = sensor_msgs.CameraInfo(width=256, height=128,
                                 k=[128., 0., 127.5, 0., 96., 63.5, 0., 0., 1.],
                                 binning_x=binning[0], binning_y=binning[1])
    info.roi = sensor_msgs.RegionOfInterest(x_offset=32, y_offset=8, width=192, height=96)
    original = copy.deepcopy(info)
    intr = ros_msgs.camera_info_to_intrinsics(info)
    bx, by = max(1, binning[0]), max(1, binning[1])
    assert (intr.width, intr.height) == (192 // bx, 96 // by)
    depth = np.zeros((intr.height, intr.width))
    depth[7, 11] = 4.
    points = back_project_depth(intr.K, depth)
    # The cropped pixel must back-project along its original full-sensor ray.
    expected = [[(32 + 11 * bx - 127.5) / 128 * 4, (8 + 7 * by - 63.5) / 96 * 4, 4.]]
    np.testing.assert_allclose(points, expected)
    assert info == original


def test_full_camera_info_roi_and_default_roi_are_equivalent():
    info = sensor_msgs.CameraInfo(width=256, height=128, binning_x=2, binning_y=2,
                                 k=[128., 0., 127.5, 0., 96., 63.5, 0., 0., 1.])
    default = ros_msgs.camera_info_to_intrinsics(info)
    info.roi = sensor_msgs.RegionOfInterest(width=256, height=128)
    explicit = ros_msgs.camera_info_to_intrinsics(info)
    assert default == explicit
    assert (explicit.width, explicit.height, explicit.cx, explicit.cy) == (128, 64, 63.75, 31.75)


@pytest.mark.parametrize('invalid', ['uncalibrated', 'nonfinite', 'empty', 'partial_roi', 'outside_roi', 'binning'])
def test_invalid_camera_info_is_rejected(invalid):
    info = sensor_msgs.CameraInfo(width=256, height=128, k=[128., 0., 127.5, 0., 96., 63.5, 0., 0., 1.])
    if invalid == 'uncalibrated':
        info.k[0] = 0.
    elif invalid == 'nonfinite':
        info.k[2] = np.nan
    elif invalid == 'empty':
        info.width = 0
    elif invalid == 'partial_roi':
        info.roi.x_offset = 32
    elif invalid == 'outside_roi':
        info.roi = sensor_msgs.RegionOfInterest(x_offset=250, width=20, height=30)
    else:
        info.binning_y = 256
    with pytest.raises(ValueError, match='CameraInfo'):
        ros_msgs.camera_info_to_intrinsics(info)
