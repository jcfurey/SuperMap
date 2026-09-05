"""Helpers for the ROS 2 end-to-end tests: build real messages from synthetic
frames and hand them to the node's synchronized callback.

Imported only by the test modules, which are not collected without ROS 2
(see conftest.py), so the ROS imports here are safe.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header

from semantic_mapping.detectors.base import Detector
from semantic_mapping.geometry_utils import (
    back_project_depth, rotation_matrix_to_quaternion, transform_points,
)
from semantic_mapping.ros_msgs import numpy_to_image

ROOT = Path(__file__).resolve().parents[2]


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
    spin_until(node, lambda: not node._pending_frames)


def spin_until(node, predicate, timeout: float = 5.0) -> None:
    """Wait through the real executor, so result draining cannot depend on test-only calls."""
    import rclpy

    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.02)
    assert predicate(), 'node did not reach the expected state before the deadline'
