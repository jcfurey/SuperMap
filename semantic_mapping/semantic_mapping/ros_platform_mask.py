"""ROS side of the platform mask: ``robot_description`` in, TF poses of its links.

Shared by the object node and the dense-cloud node; the geometry and the
rendering live in :mod:`semantic_mapping.platform_mask`.
"""
from __future__ import annotations

import threading
from collections.abc import Mapping

import numpy as np
import rclpy.time
import tf2_ros
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from semantic_mapping.platform_mask import GEOMETRY_KINDS, PlatformModel
from semantic_mapping.ros_msgs import transform_to_se3
from semantic_mapping.ros_node_utils import declare
from semantic_mapping.types import CameraIntrinsics

_DESCRIPTION_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)
"""Matches robot_state_publisher's latched robot_description."""


class PlatformUnavailable(Exception):
    """Masking is enabled but no usable robot_description has arrived yet."""


def declare_platform_mask_parameters(node) -> None:
    d = declare
    d(node, "platform_mask.enabled", False,
      "Exclude the robot's own body (robot_description links at their TF poses) from camera detections.")
    d(node, "platform_mask.robot_description_topic", "robot_description", "URDF topic (latched std_msgs/String).")
    d(node, "platform_mask.geometry", "visual",
      "visual | collision; a link without loadable geometry of this kind uses the other.")
    d(node, "platform_mask.mesh_paths", [],
      "Directories searched for package://pkg/... meshes (<dir>/pkg or <dir>/share/pkg) before the ament index.",
      dynamic_typing=True)
    d(node, "platform_mask.mesh_resolution_m", 0.02, "Simplify meshes by merging vertices in cells of this size; 0 keeps them.",
      range=(0.0, 1.0))
    d(node, "platform_mask.near_clip_m", 0.15,
      "Ignore geometry nearer than this to the image plane (the camera's own housing and lens).", range=(1e-3, 10.0))
    d(node, "platform_mask.padding_px", 8, "Dilate the platform mask by this many pixels.", range=(0, 256))
    d(node, "platform_mask.exclude_links", [], "Links left out of the mask.", dynamic_typing=True)
    d(node, "platform_mask.frame_prefix", "", "TF frame of a link: this prefix + the link name.")
    d(node, "platform_mask.max_overlap", 0.5,
      "Drop a detection with more than this fraction of its mask (box without one) on the platform; "
      "others lose their platform pixels.", range=(0.0, 1.0))


def _string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [str(item) for item in value]


class PlatformMaskSource:
    """Builds the platform model from ``robot_description`` and poses its links through TF."""

    def __init__(self, node) -> None:
        self._node = node
        param = lambda name: node.get_parameter(f"platform_mask.{name}").value  # noqa: E731
        self.enabled = bool(param("enabled"))
        self.max_overlap = float(param("max_overlap"))
        self.topic = str(param("robot_description_topic"))
        self._geometry = str(param("geometry"))
        if self._geometry not in GEOMETRY_KINDS:
            raise ValueError(f"platform_mask.geometry must be one of {GEOMETRY_KINDS}")
        self._mesh_paths = _string_list(param("mesh_paths"))
        self._exclude = _string_list(param("exclude_links"))
        self._mesh_resolution = float(param("mesh_resolution_m"))
        self._near_clip = float(param("near_clip_m"))
        self._padding = int(param("padding_px"))
        self._prefix = str(param("frame_prefix"))
        self.model: PlatformModel | None = None
        self._warned: set[tuple[str, str]] = set()
        self._lock = threading.Lock()
        self._subscription = None
        if self.enabled:
            if not self.topic:
                raise ValueError("platform_mask.robot_description_topic must be set when platform_mask.enabled")
            self._subscription = node.create_subscription(
                String, self.topic, self._on_description, _DESCRIPTION_QOS,
                callback_group=MutuallyExclusiveCallbackGroup())

    def destroy(self) -> None:
        if self._subscription is not None:
            self._node.destroy_subscription(self._subscription)
            self._subscription = None

    def _on_description(self, msg: String) -> None:
        logger = self._node.get_logger()
        try:
            model = PlatformModel.from_urdf(
                msg.data, geometry=self._geometry, mesh_paths=self._mesh_paths,
                mesh_resolution_m=self._mesh_resolution, exclude_links=self._exclude,
                padding_px=self._padding, near_clip_m=self._near_clip)
        except Exception as exc:  # noqa: BLE001 - a bad description must not stop the node
            logger.error(f"platform mask: robot_description on {self.topic} unusable, "
                         f"{'keeping the previous model' if self.model else 'frames wait'}: {exc}")
            return
        for message in model.skipped:
            logger.warning(f"platform mask: {message}")
        with self._lock:
            self.model = model
            self._warned.clear()
        logger.info(f"platform mask: {len(model.links)} links, {model.triangle_count} triangles "
                    f"({self._geometry} geometry) from {self.topic}")

    def poses(self, tf_buffer, camera_frame: str, stamp, timeout=None) -> tuple[PlatformModel, dict[str, np.ndarray]]:
        """Camera-from-link transform of every masked link at ``stamp``.

        Raises :class:`PlatformUnavailable` before a description arrived and
        ``tf2_ros.TransformException`` when a link's TF is missing.
        """
        model = self.model
        if model is None:
            raise PlatformUnavailable(f"waiting for robot_description on {self.topic}")
        time = rclpy.time.Time.from_msg(stamp) if not isinstance(stamp, rclpy.time.Time) else stamp
        extra = {} if timeout is None else {"timeout": timeout}
        poses = {}
        for link in model.links:
            try:
                poses[link] = transform_to_se3(tf_buffer.lookup_transform(camera_frame, self._prefix + link, time, **extra))
            except tf2_ros.TransformException as exc:
                # robot_state_publisher broadcasts a movable link only once its joint has a state.
                raise type(exc)(f"platform link {self._prefix + link}: {exc} (publish its joint state or list it "
                                "in platform_mask.exclude_links)") from exc
        return model, poses

    def render(self, model: PlatformModel, poses: Mapping[str, np.ndarray], intrinsics: CameraIntrinsics,
               camera_frame: str) -> np.ndarray:
        mask = model.render(poses, intrinsics, camera_frame)
        new = model.inside_warnings - self._warned
        if new:
            with self._lock:
                self._warned |= new
            links = ", ".join(sorted(link for _, link in new))
            self._node.get_logger().warning(
                f"platform mask: {camera_frame} lies inside {links}; that geometry is left out of its mask "
                "(list the camera's own housing in platform_mask.exclude_links to silence this)")
        return mask

    def mask(self, tf_buffer, camera_frame: str, stamp, intrinsics: CameraIntrinsics, timeout=None) -> np.ndarray:
        model, poses = self.poses(tf_buffer, camera_frame, stamp, timeout)
        return self.render(model, poses, intrinsics, camera_frame)
