"""End-to-end checks of the ROS 2 node and the bag converter.

These need a sourced ROS 2 environment (rclpy, sensor_msgs_py, tf2_ros,
rosbag2_py) and are skipped anywhere else; the Docker CI job runs them
inside the Jazzy image. Frames of the synthetic scene are turned into real
messages and handed to the node's synchronized callback directly, so no
topics, bags, or executors are involved and each test runs in a few seconds.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")

from builtin_interfaces.msg import Time as TimeMsg  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from sensor_msgs.msg import CameraInfo  # noqa: E402
from sensor_msgs_py import point_cloud2 as pc2  # noqa: E402
from std_msgs.msg import Header  # noqa: E402

from semantic_mapping.datasets import SequenceDataset  # noqa: E402
from semantic_mapping.detectors.base import Detector  # noqa: E402
from semantic_mapping.geometry_utils import (  # noqa: E402
    back_project_depth, rotation_matrix_to_quaternion, transform_points,
)
from semantic_mapping.ros_msgs import numpy_to_image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def scene_dir(tmp_path_factory) -> Path:
    """The synthetic scene: the checked-out one if it was generated, else a fresh one in a temp dir."""
    existing = ROOT / "data" / "example_scene"
    if (existing / "intrinsics.json").exists():
        return existing
    out = tmp_path_factory.mktemp("scene") / "example_scene"
    subprocess.run(
        [sys.executable, str(ROOT / "examples" / "prepare_example_dataset.py"), "--out_dir", str(out)],
        check=True, capture_output=True, text=True,
    )
    return out


@pytest.fixture(scope="session")
def dataset(scene_dir) -> SequenceDataset:
    return SequenceDataset(scene_dir)


class ReplayDetector(Detector):
    """Wraps the node's detector to record which frames the detector thread processed."""

    def __init__(self, inner: Detector) -> None:
        self.inner = inner
        self.calls: list[int | None] = []

    def detect(self, rgb_image, prompts=None, **kwargs):
        self.calls.append(kwargs.get("frame_id"))
        return self.inner.detect(rgb_image, prompts=prompts, **kwargs)


def stamp_msg(t: float) -> TimeMsg:
    return TimeMsg(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))


def camera_info(intr, header: Header) -> CameraInfo:
    return CameraInfo(header=header, width=intr.width, height=intr.height,
                      k=[intr.fx, 0.0, intr.cx, 0.0, intr.fy, intr.cy, 0.0, 0.0, 1.0])


def set_camera_tf(node, frame, world_frame: str = "map") -> None:
    """Inject world -> camera for this frame into the node's TF buffer."""
    tf = TransformStamped()
    tf.header = Header(stamp=stamp_msg(frame.stamp), frame_id=world_frame)
    tf.child_frame_id = node.camera_frame
    t = frame.T_world_from_cam[:3, 3]
    q = rotation_matrix_to_quaternion(frame.T_world_from_cam[:3, :3])
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, t)
    tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = map(float, q)
    node.tf_buffer.set_transform(tf, "test")


def world_points(dataset, frame, point_fraction: float | None = None, rng=None) -> np.ndarray:
    points = transform_points(frame.T_world_from_cam, back_project_depth(dataset.intrinsics.K, frame.depth))
    if point_fraction is not None:
        rng = rng or np.random.default_rng(0)
        points = points[rng.random(points.shape[0]) < point_fraction]
    return points


def feed_frame(node, dataset, frame, points_world: np.ndarray | None = None) -> None:
    """Build RGB, CameraInfo, PointCloud2 (in the map frame), and Odometry for one frame and hand them to the node."""
    set_camera_tf(node, frame)
    header = Header(stamp=stamp_msg(frame.stamp), frame_id=node.camera_frame)
    rgb_msg = numpy_to_image(frame.rgb, "rgb8", header)
    info = camera_info(dataset.intrinsics, header)
    if points_world is None:
        points_world = world_points(dataset, frame)
    cloud = pc2.create_cloud_xyz32(Header(stamp=stamp_msg(frame.stamp), frame_id="map"), points_world.tolist())
    odom = Odometry(header=Header(stamp=stamp_msg(frame.stamp), frame_id="map"), child_frame_id="sensor")
    node._on_synced_frame(rgb_msg, info, cloud, odom)


def feed_frames(node, dataset, frames, point_fraction: float | None = None, sleep: float = 0.15) -> None:
    """Feed frames in order, giving the detector thread time to answer, then fuse what it returned."""
    rng = np.random.default_rng(0)
    for frame in frames:
        feed_frame(node, dataset, frame, world_points(dataset, frame, point_fraction, rng))
        time.sleep(sleep)
    node._drain_detection_results(Header(stamp=stamp_msg(frames[-1].stamp), frame_id="map"))


@pytest.fixture
def node_factory(dataset):
    """``make(*ros_args)`` builds a node with the offline detector replaying the scene's detections.

    Parameters are given as ``-p name:=value`` pairs. Making a second node in
    the same test tears the first one down, since parameters travel through
    ``rclpy.init``. Publishers are silenced unless ``silence_publishers=False``.
    """
    from semantic_mapping.node import SemanticMappingNode

    nodes: list = []

    def make(*ros_args: str, silence_publishers: bool = True):
        if rclpy.ok():
            for node in nodes:
                node.destroy_node()
            nodes.clear()
            rclpy.shutdown()
        rclpy.init(args=[
            "--ros-args", "-p", "detector:=offline", "-p", f"offline.detections_dir:={dataset.detections_dir}",
            "-p", f"prompts_file:={ROOT / 'config' / 'prompts.yaml'}", "-p", "detector_rate_hz:=100.0",
            *ros_args,
        ])
        node = SemanticMappingNode()
        if silence_publishers:
            for publisher in (node.obj_boxes_pub, node.obj_points_pub, node.annotated_image_pub):
                publisher.publish = lambda message: None
        nodes.append(node)
        return node

    yield make
    for node in nodes:
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
