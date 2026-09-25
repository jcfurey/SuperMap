"""Live ROS2 entry point.

Subscribes to synchronized RGB, CameraInfo, PointCloud2 (or an aligned depth
image), and Odometry topics produced by an upstream geometric SLAM backbone
(see Sec. IV-A), runs the asynchronous open-vocabulary detector at its own
rate, and drives :class:`semantic_mapping.pipeline.SemanticMappingPipeline` on
every synchronized frame. Publishes per-object voxels, labeled 3D boxes (as
RViz markers and as ``vision_msgs/Detection3DArray``), and an annotated debug
image, as documented in the project README.

The node is a managed (lifecycle) node. With ``autostart`` (the default) it
configures and activates itself, so a plain ``ros2 run`` or launch ``Node``
works; set ``autostart:=false`` to hand it to ``nav2_lifecycle_manager``.
Models load in ``on_configure``; sensor input is processed only while active.

Camera/LiDAR extrinsics and the world-from-camera pose (Eq. 3) are resolved
through TF2 rather than a single hardcoded extrinsic parameter: different
SLAM backbones publish their point cloud in different frames (world-registered
vs. sensor-frame) and provide the pose in different ways, so composing it by
hand from the Odometry message would only be correct for one specific
backbone/topology. Looking up ``world_frame -> camera_frame`` and
``camera_frame -> <point cloud frame>`` through the TF tree instead works
with any upstream SLAM backbone and any (URDF, robot_state_publisher, or
static_transform_publisher) source of the camera extrinsic. A frame whose
transforms have not arrived yet waits up to ``tf_wait_sec`` before it is
dropped.

Threading: the node runs under a MultiThreadedExecutor. Sensor input, result
draining and publishing share one mutually exclusive callback group; map
save/load and the grounding action each have their own group. Every access to
the pipeline and the frame bookkeeping holds ``self._lock``. The detector and
the VLM run in worker threads that never touch the pipeline.
"""
from __future__ import annotations

import copy
import json
import math
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path as FilePath

import numpy as np
import rclpy
import rclpy.time
import tf2_ros
import yaml
from diagnostic_msgs.msg import DiagnosticStatus
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry, Path
from rcl_interfaces.msg import SetParametersResult
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType, JumpThreshold
from rclpy.duration import Duration
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import ColorRGBA, Header, String
from supermap_msgs.action import GroundInstruction
from supermap_msgs.msg import CameraAnnotations
from supermap_msgs.srv import LoadMap, SaveMap
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from semantic_mapping.detectors import build_detector
from semantic_mapping.geometry_utils import invert_se3, rasterize_depth, transform_points
from semantic_mapping.pipeline import FrameResult, PipelineConfig, SemanticMappingPipeline
from semantic_mapping.ros_msgs import (
    camera_info_has_distortion, camera_info_to_intrinsics, depth_image_to_meters, image_to_numpy, numpy_to_image,
    pointcloud_to_xyz, rectified_camera_info, transform_to_se3,
)
from semantic_mapping.ros_node_utils import (
    AutostartLifecycleNode, TransitionCallbackReturn, declare, run_node, stable_label_color,
)
from semantic_mapping.types import CameraIntrinsics, Detection2D, ObjectStatus, Observation, StampedPose
from semantic_mapping.vln.clients import build_vlm_client
from semantic_mapping.vln.grounding import Grounder, GroundingRequest, GroundingResult

PACKAGE_NAME = "semantic_mapping"

_POINT_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
"""Memory layout of one published object point; must match _POINT_FIELDS."""

_POINT_FIELDS = [
    pc2.PointField(name="x", offset=0, datatype=pc2.PointField.FLOAT32, count=1),
    pc2.PointField(name="y", offset=4, datatype=pc2.PointField.FLOAT32, count=1),
    pc2.PointField(name="z", offset=8, datatype=pc2.PointField.FLOAT32, count=1),
    pc2.PointField(name="rgb", offset=12, datatype=pc2.PointField.FLOAT32, count=1),
]

_MAP_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)
"""Latched map outputs: a late RViz or planner gets the current map immediately."""

_IMAGE_QOS = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
"""Debug images: reliable (matches RViz defaults) but shallow, so a slow viewer never queues frames."""

_MAX_SUPPORT_EXTENT_M = 4.0
"""Obstacles wider than this (floors, rooms, huge clusters) are ignored when choosing an approach pose."""

_MAX_HYPOTHESES = 5
"""Label hypotheses reported per object in Detection3DArray, most likely first."""


def _label_color(label: str) -> tuple[float, float, float]:
    """Deterministic RGB for a label; the same in every process and in dense_yoloe_node."""
    return stable_label_color(label)


def _packed_rgb_float(label: str) -> np.float32:
    """The label's colour packed as PCL's float-typed rgb field (0x00RRGGBB reinterpreted as float32)."""
    r, g, b = _label_color(label)
    packed = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
    return np.frombuffer(np.array([packed], dtype=np.uint32).tobytes(), dtype=np.float32)[0]


def _stamp_to_seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def _advance_schedule(previous: float, stamp: float, period: float) -> float:
    """Retain the rate's phase across input jitter; skip missed slots without bursts."""
    if not np.isfinite(previous):
        return stamp
    return min(stamp, previous + max(1, int((stamp - previous) / period)) * period)


def _rate_due(previous: float, stamp: float, period: float) -> bool:
    # Camera timestamps jitter around their nominal period. Allow at most
    # 1 ms (and at most 1% of a period) early, instead of dropping a whole
    # camera frame for a sub-millisecond difference at the requested rate.
    return stamp - previous >= period - min(0.001, 0.01 * period)


def _yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def _footprint(bbox) -> np.ndarray:
    """[xmin, ymin, xmax, ymax] of an axis-aligned 3D box."""
    return np.array([bbox[0], bbox[1], bbox[3], bbox[4]], dtype=np.float64)


def _overlaps(a: np.ndarray, b: np.ndarray) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _ray_distance_for_clearance(half: np.ndarray, direction: np.ndarray, standoff: float) -> float:
    """Distance along ``direction`` from a box centre to the point ``standoff`` metres from the box."""
    def gap(s: float) -> float:
        return float(np.hypot(*np.maximum(np.abs(s * direction) - half, 0.0)))

    lo = min(half[i] / abs(direction[i]) for i in range(2) if abs(direction[i]) > 1e-9)
    if standoff <= 0:
        return lo
    hi = lo + standoff
    while gap(hi) < standoff:
        hi += standoff
    for _ in range(60):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if gap(mid) < standoff else (lo, mid)
    return hi


def approach_pose(bbox, robot_xy=None, standoff: float = 0.6, obstacles=(), clearance: float = 0.2,
                  num_directions: int = 16) -> tuple[float, float, float]:
    """A reachable goal next to an object rather than its (occupied) centroid.

    Returns ``(x, y, yaw)``: a point ``standoff`` metres outside the object's
    footprint, facing the object's centre. The approach direction is the one
    towards ``robot_xy`` when it is known; otherwise the object's narrowest
    side is tried first, so the goal is as close to the object as possible.
    A candidate inside another object's footprint (grown by ``clearance``) is
    rejected in favour of the next direction around the object. Objects whose
    footprint overlaps the target (a mug's table) are merged into it, so the
    goal lands beside the supporting furniture instead of on it. Obstacles
    wider than ``_MAX_SUPPORT_EXTENT_M`` (floors) are ignored. If every
    direction is blocked, the first candidate is returned.
    """
    target = _footprint(bbox)
    center = np.array([(target[0] + target[2]) / 2.0, (target[1] + target[3]) / 2.0])
    blockers = []
    for other in obstacles:
        fp = _footprint(other)
        if max(fp[2] - fp[0], fp[3] - fp[1]) > _MAX_SUPPORT_EXTENT_M:
            continue
        if _overlaps(fp, target):
            target = np.array([min(target[0], fp[0]), min(target[1], fp[1]),
                               max(target[2], fp[2]), max(target[3], fp[3])])
        else:
            blockers.append(fp + np.array([-clearance, -clearance, clearance, clearance]))
    box_center = np.array([(target[0] + target[2]) / 2.0, (target[1] + target[3]) / 2.0])
    half = np.array([(target[2] - target[0]) / 2.0, (target[3] - target[1]) / 2.0])

    step = 2.0 * math.pi / max(int(num_directions), 4)
    if robot_xy is not None and np.linalg.norm(np.asarray(robot_xy, float)[:2] - box_center) > 1e-6:
        offset = np.asarray(robot_xy, float)[:2] - box_center
        start = math.atan2(offset[1], offset[0])
        order = [0]
        for k in range(1, int(num_directions) // 2 + 1):
            order += [k, -k]
        angles = [start + k * step for k in order][:int(num_directions)]
    else:
        # Narrowest side first: +-x when the box is thinner in x, else +-y, then the rest.
        faces = [0.0, math.pi, math.pi / 2, -math.pi / 2] if half[0] <= half[1] else \
            [math.pi / 2, -math.pi / 2, 0.0, math.pi]
        angles = faces + [k * step for k in range(int(num_directions)) if not any(
            abs(math.remainder(k * step - f, 2 * math.pi)) < 1e-9 for f in faces)]

    first = None
    for angle in angles:
        direction = np.array([math.cos(angle), math.sin(angle)])
        goal = box_center + direction * _ray_distance_for_clearance(half, direction, standoff)
        yaw = math.atan2(center[1] - goal[1], center[0] - goal[0])
        candidate = (float(goal[0]), float(goal[1]), float(yaw))
        if first is None:
            first = candidate
        if not any(b[0] <= goal[0] <= b[2] and b[1] <= goal[1] <= b[3] for b in blockers):
            return candidate
    return first


class _ScanRing:
    """The last ``maxlen`` world-frame scans in one preallocated buffer.

    Each scan is transformed into the world once, on arrival. Unused rows hold
    NaN, which :func:`rasterize_depth` discards, so the whole buffer can be
    reprojected under the current camera pose without re-concatenating the
    scans on every frame.
    """

    def __init__(self, maxlen: int) -> None:
        self.maxlen = max(int(maxlen), 1)
        self._buffer = np.full((self.maxlen, 0, 3), np.nan)
        self._counts = [0] * self.maxlen
        self._next = 0
        self._size = 0

    def append(self, points: np.ndarray) -> None:
        n = int(points.shape[0])
        capacity = self._buffer.shape[1]
        if n > capacity:
            grown = np.full((self.maxlen, max(n, int(capacity * 1.25) + 1), 3), np.nan)
            grown[:, :capacity] = self._buffer
            self._buffer = grown
        slot = self._next
        self._buffer[slot, :n] = points
        self._buffer[slot, n:] = np.nan
        self._counts[slot] = n
        self._next = (slot + 1) % self.maxlen
        self._size = min(self._size + 1, self.maxlen)

    def points(self) -> np.ndarray:
        """Every stored point (with NaN padding rows), as one (M, 3) view."""
        return self._buffer.reshape(-1, 3)

    def clear(self) -> None:
        self._buffer[:] = np.nan
        self._counts = [0] * self.maxlen
        self._next = self._size = 0

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> np.ndarray:
        """Scan ``index``, oldest first."""
        if not -self._size <= index < self._size:
            raise IndexError(index)
        slot = (self._next - self._size + index % self._size) % self.maxlen
        return self._buffer[slot, :self._counts[slot]]


@dataclass
class _PendingFrame:
    observation: Observation
    header: Header
    detector_started_at: float | None = None
    annotate: bool = False
    camera_info: CameraInfo | None = None
    """Rectified calibration of this frame, kept only for the dense annotation export."""


@dataclass
class _WaitingFrame:
    """Synchronized input waiting for its transforms to arrive."""

    rgb: object
    info: CameraInfo
    depth: object
    stamp: float
    arrived: float


@dataclass
class _GroundingOutcome:
    success: bool
    message: str
    target_ids: list[int] = field(default_factory=list)
    target_labels: list[str] = field(default_factory=list)
    unresolved_ids: list[int] = field(default_factory=list)
    goals: list[tuple[float, float, float, float]] = field(default_factory=list)
    """Approach poses ``(x, y, z, yaw)`` in the world frame, one per resolved target."""
    response: str = ""
    result: GroundingResult | None = None


@dataclass
class _GroundingJob:
    request_id: str
    instruction: str
    request: GroundingRequest | None
    bboxes: dict[int, np.ndarray]
    labels: dict[int, str]
    obstacles: dict[int, np.ndarray]
    robot_xy: np.ndarray | None
    goal_z: float
    feedback: object = None
    """Callable(stage) publishing action feedback, or None for topic queries."""
    on_done: object = None
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: bool = False
    outcome: _GroundingOutcome | None = None


class _GroundingUnavailable(Exception):
    pass


# Parameters that ``ros2 param set`` may change while running, with their validators.
_TUNABLE = {
    "detector_rate_hz": lambda v: v > 0 and math.isfinite(v),
    "publish_rate_hz": lambda v: v > 0 and math.isfinite(v),
    "detector_timeout_sec": lambda v: v > 0 and math.isfinite(v),
    "max_pending_frames": lambda v: v >= 1,
    "publish_disappeared_objects": lambda v: True,
    "tf_wait_sec": lambda v: v >= 0 and math.isfinite(v),
    "input_time_jump_reset_sec": lambda v: v >= 0 and math.isfinite(v),
    "vlm.local_radius_m": lambda v: v >= 0 and math.isfinite(v),
    "vlm.max_objects": lambda v: v >= 0,
    "vlm.stale_after_sec": lambda v: v >= 0 and math.isfinite(v),
    "grounding_max_queue": lambda v: v >= 1,
    "goal_standoff_m": lambda v: v >= 0 and math.isfinite(v),
    "goal_clearance_m": lambda v: v >= 0 and math.isfinite(v),
    "goal_use_robot_z": lambda v: True,
    "goal_z_m": lambda v: math.isfinite(v),
    "nav2_send_goal": lambda v: True,
}


class SemanticMappingNode(AutostartLifecycleNode):
    def __init__(self, **kwargs) -> None:
        super().__init__("semantic_mapping_node", **kwargs)
        self._lock = threading.RLock()
        """Guards the pipeline, the frame bookkeeping and the last result."""
        self._stats_lock = threading.Lock()
        self._guard_lock = threading.Lock()
        self._slot_lock = threading.Lock()
        self._configured = False
        self._active = False
        self._updater = None
        self._declare_parameters()
        self.add_on_set_parameters_callback(self._on_set_parameters)

    # ------------------------------------------------------------ parameters
    def _declare_parameters(self) -> None:
        d = self._declare
        # Topics are relative so the node can be namespaced; remap them or set these.
        d("rgb_topic", "camera/color/image_raw", "RGB image (sensor_msgs/Image or CompressedImage).")
        d("camera_info_topic", "camera/color/camera_info", "CameraInfo of the RGB stream.")
        d("pointcloud_topic", "lidar/points", "PointCloud2 used as depth when depth_source=pointcloud.")
        d("odometry_topic", "odometry", "Odometry that paces processing when sync_odometry=true.")
        d("sync_odometry", True, "Also synchronize on Odometry; poses always come from TF.")
        d("depth_source", "pointcloud", "pointcloud | depth_image (aligned to the RGB camera).")
        d("depth_topic", "camera/aligned_depth_to_color/image_raw", "Aligned depth image topic.")
        d("depth_scale", 1000.0, "Units per metre of 16-bit depth images.")
        d("pointcloud_accumulate_scans", 1, "Rasterize the last N scans (via TF) for sparse LiDAR.",
          range=(1, 100))
        d("rgb_compressed", False, "rgb_topic carries sensor_msgs/CompressedImage.")
        d("sensor_qos", "best_effort", "best_effort | reliable | sensor_data.")
        d("sensor_qos_depth", 10, "History depth of sensor subscriptions.", range=(1, 1000))
        d("sync_slop_sec", 0.05, "Approximate-time synchronizer slop.")
        d("sync_queue_size", 30, "Approximate-time synchronizer queue.", range=(1, 1000))
        d("camera_images_rectified", False,
          "Images are rectified (image_proc image_rect): project with CameraInfo P and ignore D. "
          "false: images are raw and CameraInfo D must be zero; K is used.")

        d("obj_points_topic", "obj_points", "Label-coloured object points (latched).")
        d("obj_boxes_topic", "obj_boxes", "RViz MarkerArray of object boxes and labels (latched).")
        d("objects_topic", "objects", "vision_msgs/Detection3DArray of mapped objects (latched).")
        d("annotated_image_topic", "semantic_mapping/annotated_image", "Detections drawn on the RGB image.")
        d("publish_dense_camera_annotations", False, "Publish YOLOE masks as supermap_msgs/CameraAnnotations.")
        d("dense_camera_images_rectified", False, "Deprecated alias of camera_images_rectified.")
        d("dense_camera_annotations_topic", "supermap/camera_annotations", "CameraAnnotations output topic.")

        d("world_frame", "map", "Fixed frame of the map and all outputs.")
        d("camera_frame", "camera_color_optical_frame", "Optical frame of the RGB camera.")
        d("robot_frame", "", "Robot base frame for grounding (distance, goal height). Empty: camera_frame.")
        d("tf_wait_sec", 0.2, "How long a frame waits for its transforms before it is dropped.",
          read_only=False)
        d("input_time_jump_reset_sec", 1.0,
          "A clock or input-stamp jump backwards by more than this resets the input state (bag loops). "
          "0 disables.", read_only=False)

        d("detector", "offline", "offline | yoloe | groundingdino.")
        d("detector_rate_hz", 1.0, "Detector rate.", read_only=False)
        d("detector_timeout_sec", 2.0, "Deadline for a detection; later results are discarded.",
          read_only=False)
        d("max_pending_frames", 30, "Frames waiting behind a detection before they are released.",
          read_only=False)
        d("prompts_file", "config/prompts.yaml",
          "Open-vocabulary prompt list. Relative paths resolve against the package share directory.")
        d("publish_rate_hz", 5.0, "Map output rate.", read_only=False)
        d("publish_disappeared_objects", True, "Also draw retired instances.", read_only=False)
        d("stats_log_period_sec", 10.0, "Log module rates every N s; 0 disables.")

        d("query_topic", "semantic_mapping/query", "Legacy std_msgs/String instruction input.")
        d("answer_topic", "semantic_mapping/answer", "Legacy JSON answer output (with request_id).")
        d("goal_topic", "semantic_mapping/goal", "First approach pose of the latest grounding.")
        d("waypoints_topic", "semantic_mapping/waypoints", "All approach poses of the latest grounding.")
        d("grounding_max_queue", 4, "Grounding requests queued or running; more are rejected.",
          read_only=False)
        d("goal_standoff_m", 0.6, "Distance of an approach pose from the object's footprint.",
          read_only=False)
        d("goal_clearance_m", 0.2, "Clearance of an approach pose from other objects.", read_only=False)
        d("goal_use_robot_z", True, "Approach pose height from robot_frame; goal_z_m when unknown.",
          read_only=False)
        d("goal_z_m", 0.0, "Approach pose height when not taken from the robot.", read_only=False)
        d("nav2_send_goal", False, "Send the first approach pose to Nav2 NavigateToPose.", read_only=False)
        d("nav2_action_name", "navigate_to_pose", "Nav2 NavigateToPose action name.")
        d("vlm.client", "keyword", "keyword | openai_compatible | anthropic.")
        d("vlm.model", "", "Model name for the VLM backend.")
        d("vlm.base_url", "", "Endpoint of an OpenAI-compatible backend.")
        d("vlm.api_key_env", "", "Environment variable holding the API key.")
        d("vlm.max_tokens", 0, "Response token budget, thinking included (0: backend default).")
        d("vlm.max_retries", -1, "Retries on timeouts / 429 / 5xx (-1: backend default).")
        d("vlm.retry_backoff_s", 0.0, "Initial retry backoff in seconds (0: backend default).")
        d("vlm.local_radius_m", 0.0, "> 0: serialize only objects this close to the robot.", read_only=False)
        d("vlm.max_objects", 0, "> 0: cap the serialized objects (nearest first).", read_only=False)
        d("vlm.stale_after_sec", 30.0, "Occluded instances unseen longer than this are marked stale.",
          read_only=False)

        d("map_load_path", "", "Map directory restored on configure and by ~/load_map with an empty path.")
        d("map_save_path", "", "Map directory for ~/save_map with an empty path and for autosave.")
        d("map_autosave_sec", 0.0, "> 0: save to map_save_path every N seconds.")

        for name, value in PipelineConfig().__dict__.items():
            d(name, value, f"PipelineConfig.{name}; see config/semantic_mapping.yaml.")
        d("yoloe.checkpoint", "yoloe-v8l-seg.pt", "YOLOE checkpoint.")
        d("yoloe.device", "cuda", "YOLOE device.")
        d("yoloe.confidence_threshold", 0.25, "YOLOE confidence threshold.")
        d("yoloe.text_encoder_path", "", "Local MobileCLIP TorchScript asset (dense export).")
        d("yoloe.sam2_checkpoint", "", "Optional SAM2 refinement checkpoint.")
        d("yoloe.sam2_model_cfg", "", "Optional SAM2 model config.")
        d("groundingdino.config_path", "", "GroundingDINO config.")
        d("groundingdino.checkpoint_path", "", "GroundingDINO checkpoint.")
        d("groundingdino.sam2_checkpoint", "", "Optional SAM2 refinement checkpoint.")
        d("groundingdino.sam2_model_cfg", "", "Optional SAM2 model config.")
        d("groundingdino.device", "cuda", "GroundingDINO device.")
        d("groundingdino.box_threshold", 0.35, "GroundingDINO box threshold.")
        d("groundingdino.text_threshold", 0.25, "GroundingDINO text threshold.")
        d("offline.detections_dir", "", "Directory of pre-computed detections (offline detector).")

    def _declare(self, name, default, description="", *, read_only=True, range=None):
        return declare(self, name, default, description, read_only=read_only, range=range)

    def _param_str(self, name: str, default: str = "") -> str:
        value = self.get_parameter(name).value
        return default if value is None else str(value)

    def _param(self, name: str):
        return self.get_parameter(name).value

    def _on_set_parameters(self, params) -> SetParametersResult:
        values = {}
        for param in params:
            if param.name not in _TUNABLE:
                continue  # read-only parameters are rejected by their descriptors
            value = param.value
            if isinstance(value, bool) != isinstance(self._param(param.name), bool):
                return SetParametersResult(successful=False, reason=f"{param.name}: wrong type")
            if not isinstance(value, bool) and isinstance(self._param(param.name), float):
                value = float(value)
            try:
                valid = _TUNABLE[param.name](value)
            except TypeError:
                valid = False
            if not valid:
                return SetParametersResult(successful=False, reason=f"{param.name}: invalid value {value!r}")
            values[param.name] = value
        if values and self._configured:
            current = {name: self._param(name) for name in _TUNABLE}
            current.update(values)
            self._apply_tunables(current)
        return SetParametersResult(successful=True)

    def _apply_tunables(self, values: dict) -> None:
        with self._lock:
            self._detector_period_sec = 1.0 / float(values["detector_rate_hz"])
            self._publish_period_sec = 1.0 / float(values["publish_rate_hz"])
            self._detector_timeout_sec = float(values["detector_timeout_sec"])
            self._max_pending_frames = int(values["max_pending_frames"])
            include_history = bool(values["publish_disappeared_objects"])
            if include_history != getattr(self, "_include_history", include_history):
                self._cloud_signature = None
            self._include_history = include_history
            self._tf_wait_sec = float(values["tf_wait_sec"])
            self._jump_reset_sec = float(values["input_time_jump_reset_sec"])
            self._grounding_max_queue = int(values["grounding_max_queue"])
            self._goal_standoff_m = float(values["goal_standoff_m"])
            self._goal_clearance_m = float(values["goal_clearance_m"])
            self._goal_use_robot_z = bool(values["goal_use_robot_z"])
            self._goal_z_m = float(values["goal_z_m"])
            self._nav2_send_goal = bool(values["nav2_send_goal"])
            if getattr(self, "grounder", None) is not None:
                self.grounder.local_radius_m = float(values["vlm.local_radius_m"]) or None
                self.grounder.max_objects = int(values["vlm.max_objects"]) or None
                self.grounder.stale_after_sec = float(values["vlm.stale_after_sec"]) or None
            if getattr(self, "_freq_bounds", None):
                rate = 1.0 / self._detector_period_sec
                self._freq_bounds["detector"].update(min=0.5 * rate, max=rate)
                self._freq_bounds["publish"].update(max=1.0 / self._publish_period_sec)

    # ------------------------------------------------------------- lifecycle
    def on_configure(self, state) -> TransitionCallbackReturn:
        try:
            self._configure()
        except Exception as exc:  # noqa: BLE001 - report and stay unconfigured
            self.get_logger().error(f"configure failed: {exc}")
            self._teardown()
            return TransitionCallbackReturn.FAILURE
        self.get_logger().info("semantic_mapping_node configured")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state) -> TransitionCallbackReturn:
        result = super().on_activate(state)
        with self._lock:
            self._active = True
            self._cloud_signature = None  # republish the latched map on (re)activation
        self.get_logger().info("semantic_mapping_node active")
        return result

    def on_deactivate(self, state) -> TransitionCallbackReturn:
        with self._lock:
            self._active = False
        return super().on_deactivate(state)

    def on_cleanup(self, state) -> TransitionCallbackReturn:
        self._teardown()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state) -> TransitionCallbackReturn:
        self._teardown()
        return TransitionCallbackReturn.SUCCESS

    def destroy_node(self) -> None:
        self._teardown()
        super().destroy_node()

    def _configure(self) -> None:
        self._stop_event = threading.Event()
        self._owned_timers: list = []
        self._subscriptions_owned: list = []
        self._publishers_owned: list = []
        self._services_owned: list = []
        self._action_server = None
        self._nav2_client = None
        self._jump_handle = None
        self._detection_ready = None
        self._detector_thread = self._grounding_thread = None

        self.world_frame = self._param_str("world_frame", "map")
        self.camera_frame = self._param_str("camera_frame", "camera_color_optical_frame")
        self.robot_frame = self._param_str("robot_frame") or self.camera_frame
        self._images_rectified = bool(self._param("camera_images_rectified")) or bool(
            self._param("dense_camera_images_rectified"))
        self.depth_from_image = self._param_str("depth_source", "pointcloud") == "depth_image"
        self.depth_scale = float(self._param("depth_scale"))
        self._apply_tunables({name: self._param(name) for name in _TUNABLE})

        detector_name = self._param_str("detector", "offline")
        self.prompts = self._load_prompts(self._param_str("prompts_file", "config/prompts.yaml"),
                                          required=detector_name != "offline")
        if detector_name != "offline" and not self.prompts:
            raise ValueError(f"detector '{detector_name}' needs a non-empty prompt vocabulary (prompts_file)")
        self._dense_annotation_model = self._configure_dense_annotations()

        self.pipeline = SemanticMappingPipeline(self._build_pipeline_config())
        self.detector = build_detector(detector_name, **self._detector_kwargs())

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._scan_history = _ScanRing(int(self._param("pointcloud_accumulate_scans")))
        self._next_frame_id = 0
        self._pending_frames: deque[_PendingFrame] = deque()
        self._pending_by_id: dict[int, _PendingFrame] = {}
        self._tf_waiting: deque[_WaitingFrame] = deque()
        self._detector_in_flight: int | None = None
        self._last_result: FrameResult | None = None
        self._published_marker_ids: set[int] = set()
        self._marker_reset_pending = True
        self._cloud_signature = None
        self._reset_input_state()

        # The worker returns independent results so a late answer cannot
        # mutate an observation that has already been committed.
        self._detection_jobs: queue.Queue[Observation] = queue.Queue(maxsize=1)
        self._detection_results: queue.Queue[tuple[int, list[Detection2D], float, bool]] = queue.Queue()

        # Language grounding (Sec. IV-D): requests snapshot and serialize the
        # graph under the lock; only the model call runs in the worker.
        self.grounder = Grounder(
            build_vlm_client(self._param_str("vlm.client", "keyword"), **self._vlm_kwargs()),
            coordinate_frame=self.world_frame,
            local_radius_m=float(self._param("vlm.local_radius_m")) or None,
            max_objects=int(self._param("vlm.max_objects")) or None,
            stale_after_sec=float(self._param("vlm.stale_after_sec")) or None,
        )
        self._grounding_jobs: queue.Queue[_GroundingJob | None] = queue.Queue()
        self._grounding_outstanding = 0
        self._query_counter = 0

        self._stats = self._empty_stats()
        self._stats_since = time.monotonic()
        self._counters = {"out_of_order": 0, "invalid_input": 0, "tf_failures": 0, "detector_timeouts": 0,
                          "time_jumps": 0, "failures": 0}
        self._failures_by_stage: dict[str, int] = {}

        self.map_save_path = self._param_str("map_save_path")
        self._sensor_group = MutuallyExclusiveCallbackGroup()
        self._io_group = MutuallyExclusiveCallbackGroup()
        self._query_group = ReentrantCallbackGroup()
        self._setup_io()

        load_path = self._param_str("map_load_path")
        if load_path:
            self.get_logger().info(self._load_map(load_path))
        autosave_sec = float(self._param("map_autosave_sec"))
        if autosave_sec > 0 and self.map_save_path:
            self._owned_timers.append(self.create_timer(autosave_sec, self._autosave, callback_group=self._io_group))
        elif autosave_sec > 0:
            self.get_logger().warning("map_autosave_sec is set but map_save_path is empty; autosave disabled")
        stats_period = float(self._param("stats_log_period_sec"))
        if stats_period > 0:
            self._owned_timers.append(self.create_timer(stats_period, self._log_runtime_stats,
                                                  callback_group=self._sensor_group))

        # Wake the executor as soon as inference finishes. Polling alone adds
        # up to 20 ms before fusion, enough to miss the next camera frame.
        self._detection_ready = self.create_guard_condition(self._drain_detection_results,
                                                            callback_group=self._sensor_group)
        # A steady-clock timer drains results, retries frames waiting for TF
        # and expires pending work even when input or a bag's /clock stops.
        self._detection_timer = self.create_timer(
            0.02, self._drain_detection_results, clock=Clock(clock_type=ClockType.STEADY_TIME),
            callback_group=self._sensor_group)
        self._owned_timers.append(self._detection_timer)
        if self._jump_reset_sec > 0:
            self._jump_handle = self.get_clock().create_jump_callback(
                JumpThreshold(min_forward=None, min_backward=Duration(seconds=-self._jump_reset_sec),
                              on_clock_change=False),
                post_callback=self._on_time_jump)
        self._setup_diagnostics()

        self._detector_thread = threading.Thread(target=self._detector_loop, name="detector", daemon=True)
        self._grounding_thread = threading.Thread(target=self._grounding_loop, name="grounding", daemon=True)
        self._detector_thread.start()
        self._grounding_thread.start()
        self._configured = True

    def _teardown(self) -> None:
        """Release everything on_configure created; safe to call repeatedly.

        Workers stop first, then the guard condition they signal is destroyed
        under the lock they trigger it with, so a detector finishing during
        shutdown can never touch a destroyed entity (C38).
        """
        if not hasattr(self, "_stop_event"):
            return
        with self._lock:
            self._active = False
            self._configured = False
        self._stop_event.set()
        if hasattr(self, "_grounding_jobs"):
            self._grounding_jobs.put(None)  # wake the grounding worker
        for worker in (self._detector_thread, self._grounding_thread):
            if worker is not None and worker is not threading.current_thread():
                worker.join(timeout=2.0)
                if worker.is_alive():
                    self.get_logger().warning(f"{worker.name} thread still busy at shutdown; it exits when done")
        self._detector_thread = self._grounding_thread = None
        with self._guard_lock:
            if self._detection_ready is not None:
                self.destroy_guard_condition(self._detection_ready)
                self._detection_ready = None
        if self._jump_handle is not None:
            self._jump_handle.unregister()
            self._jump_handle = None
        for timer in self._owned_timers:
            self.destroy_timer(timer)
        self._owned_timers = []
        if self._updater is not None:
            for name in getattr(self, "_diagnostic_task_names", []):
                self._updater.removeByName(name)
            self._diagnostic_task_names = []
            self._freq = {}
            self._updater.timer.cancel()
        if self._action_server is not None:
            self._action_server.destroy()
            self._action_server = None
        if self._nav2_client is not None:
            self._nav2_client.destroy()
            self._nav2_client = None
        for service in self._services_owned:
            self.destroy_service(service)
        self._services_owned = []
        for subscription in self._subscriptions_owned:
            self.destroy_subscription(subscription)
        self._subscriptions_owned = []
        for publisher in self._publishers_owned:
            self.destroy_lifecycle_publisher(publisher)
        self._publishers_owned = []
        if getattr(self, "tf_listener", None) is not None:
            self.tf_listener.unregister()
            self.tf_listener = None
        self.detector = None

    # ------------------------------------------------------------------ setup
    def _resolve_prompts_file(self, value: str) -> FilePath:
        """Absolute paths as given; relative ones against the package share, then the working directory."""
        path = FilePath(value).expanduser()
        if path.is_absolute():
            return path
        try:
            from ament_index_python.packages import get_package_share_directory

            shared = FilePath(get_package_share_directory(PACKAGE_NAME)) / path
            if shared.is_file():
                return shared
        except (ImportError, LookupError, ValueError):
            pass
        return path.resolve()

    def _load_prompts(self, prompts_file: str, required: bool = False) -> list[str]:
        path = self._resolve_prompts_file(prompts_file)
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as exc:
            if required:
                raise ValueError(f"cannot read prompts file '{path}': {exc}") from exc
            self.get_logger().warning(f"Could not read prompts file '{path}', using empty vocabulary")
            return []
        prompts = [str(p) for p in (data.get("prompts", []) if isinstance(data, dict) else [])]
        self.get_logger().info(f"{len(prompts)} prompts from {path}")
        return prompts

    def _build_pipeline_config(self) -> PipelineConfig:
        return PipelineConfig(**{name: self._param(name) for name in PipelineConfig().__dict__})

    def _detector_kwargs(self) -> dict:
        backend = self._param_str("detector", "offline")
        if backend == "yoloe":
            return {
                "checkpoint": self._param_str("yoloe.checkpoint", "yoloe-v8l-seg.pt"),
                "device": self._param_str("yoloe.device", "cuda"),
                "confidence_threshold": float(self._param("yoloe.confidence_threshold")),
                "text_encoder_path": self._param_str("yoloe.text_encoder_path") or None,
                "sam2_checkpoint": self._param_str("yoloe.sam2_checkpoint") or None,
                "sam2_model_cfg": self._param_str("yoloe.sam2_model_cfg") or None,
            }
        if backend in ("groundingdino", "grounding_dino", "gdino"):
            return {
                "config_path": self._param_str("groundingdino.config_path"),
                "checkpoint_path": self._param_str("groundingdino.checkpoint_path"),
                "sam2_checkpoint": self._param_str("groundingdino.sam2_checkpoint") or None,
                "sam2_model_cfg": self._param_str("groundingdino.sam2_model_cfg") or None,
                "device": self._param_str("groundingdino.device", "cuda"),
                "box_threshold": float(self._param("groundingdino.box_threshold")),
                "text_threshold": float(self._param("groundingdino.text_threshold")),
            }
        return {"detections_dir": self._param_str("offline.detections_dir")}

    def _configure_dense_annotations(self):
        if not self._param("publish_dense_camera_annotations"):
            return None
        if self._param_str("detector", "offline") != "yoloe" or self._param_str("yoloe.sam2_checkpoint"):
            raise ValueError("dense camera export supports only the explicit YOLOE exception, without extra models")
        if not (self._param("camera_images_rectified") or self._param("dense_camera_images_rectified")):
            raise ValueError("dense camera export requires rectified images (camera_images_rectified=true)")
        paths = [FilePath(self._param_str(key)).expanduser() for key in
                 ("yoloe.checkpoint", "yoloe.text_encoder_path")]
        if not all(path.is_file() for path in paths):
            raise ValueError("dense camera export requires existing local YOLOE checkpoint and text encoder")
        from semantic_mapping.dense_cloud_io import sha256_file
        digests = [sha256_file(path) for path in paths]
        return {"name": "YOLOE", "exception": "explicitly_allowed_non_US_origin",
                "checkpoint": paths[0].name, "checkpoint_sha256": digests[0],
                "text_encoder": paths[1].name, "text_encoder_sha256": digests[1]}

    def _vlm_kwargs(self) -> dict:
        """Only pass what was configured, so each backend keeps its own defaults."""
        kwargs = {}
        for key in ("model", "base_url", "api_key_env"):
            value = self._param_str(f"vlm.{key}")
            if value:
                kwargs[key] = value
        if int(self._param("vlm.max_tokens")) > 0:
            kwargs["max_tokens"] = int(self._param("vlm.max_tokens"))
        if int(self._param("vlm.max_retries")) >= 0:
            kwargs["max_retries"] = int(self._param("vlm.max_retries"))
        if float(self._param("vlm.retry_backoff_s")) > 0:
            kwargs["retry_backoff_s"] = float(self._param("vlm.retry_backoff_s"))
        return kwargs

    def _sensor_qos(self) -> QoSProfile:
        name = self._param_str("sensor_qos", "best_effort").lower()
        if name == "sensor_data":
            return qos_profile_sensor_data
        reliability = ReliabilityPolicy.RELIABLE if name == "reliable" else ReliabilityPolicy.BEST_EFFORT
        return QoSProfile(reliability=reliability, history=HistoryPolicy.KEEP_LAST,
                          depth=int(self._param("sensor_qos_depth")))

    def _publisher(self, msg_type, topic_param: str, qos):
        publisher = self.create_lifecycle_publisher(msg_type, self._param_str(topic_param), qos)
        self._publishers_owned.append(publisher)
        return publisher

    def _setup_io(self) -> None:
        qos = self._sensor_qos()
        group = self._sensor_group
        rgb_type = CompressedImage if bool(self._param("rgb_compressed")) else Image
        rgb_sub = Subscriber(self, rgb_type, self._param_str("rgb_topic"), qos_profile=qos, callback_group=group)
        info_sub = Subscriber(self, CameraInfo, self._param_str("camera_info_topic"), qos_profile=qos,
                              callback_group=group)
        if self.depth_from_image:
            depth_sub = Subscriber(self, Image, self._param_str("depth_topic"), qos_profile=qos,
                                   callback_group=group)
        else:
            depth_sub = Subscriber(self, PointCloud2, self._param_str("pointcloud_topic"), qos_profile=qos,
                                   callback_group=group)
        self._sensor_subscribers = [rgb_sub, info_sub, depth_sub]
        if bool(self._param("sync_odometry")):
            self._sensor_subscribers.append(Subscriber(
                self, Odometry, self._param_str("odometry_topic"), qos_profile=qos, callback_group=group))
        else:
            self.get_logger().info("Odometry synchronization disabled; camera poses still come from TF")
        self._subscriptions_owned.extend(s.sub for s in self._sensor_subscribers)

        self._sync = ApproximateTimeSynchronizer(
            self._sensor_subscribers,
            queue_size=int(self._param("sync_queue_size")),
            slop=float(self._param("sync_slop_sec")),
        )
        self._sync.registerCallback(self._on_synced_frame)

        self.obj_points_pub = self._publisher(PointCloud2, "obj_points_topic", _MAP_QOS)
        self.obj_boxes_pub = self._publisher(MarkerArray, "obj_boxes_topic", _MAP_QOS)
        self.objects_pub = self._publisher(Detection3DArray, "objects_topic", _MAP_QOS)
        self.annotated_image_pub = self._publisher(Image, "annotated_image_topic", _IMAGE_QOS)
        self.dense_annotations_pub = (
            self._publisher(CameraAnnotations, "dense_camera_annotations_topic", 8)
            if self._dense_annotation_model is not None else None)

        self.query_sub = self.create_subscription(
            String, self._param_str("query_topic"), self._on_query, 10, callback_group=self._query_group)
        self._subscriptions_owned.append(self.query_sub)
        self.answer_pub = self._publisher(String, "answer_topic", 10)
        self.goal_pub = self._publisher(PoseStamped, "goal_topic", _MAP_QOS)
        self.waypoints_pub = self._publisher(Path, "waypoints_topic", _MAP_QOS)

        self.save_map_srv = self.create_service(SaveMap, "~/save_map", self._on_save_map,
                                                callback_group=self._io_group)
        self.load_map_srv = self.create_service(LoadMap, "~/load_map", self._on_load_map,
                                                callback_group=self._io_group)
        self._services_owned.extend([self.save_map_srv, self.load_map_srv])
        self._action_server = ActionServer(
            self, GroundInstruction, "~/ground_instruction",
            execute_callback=self._execute_ground_instruction,
            goal_callback=self._on_ground_goal,
            cancel_callback=lambda goal_handle: CancelResponse.ACCEPT,
            callback_group=self._query_group)

    def _setup_diagnostics(self) -> None:
        from diagnostic_updater import FrequencyStatus, FrequencyStatusParam, Updater

        if self._updater is None:
            # Updater declares its own parameters, so it is created once per node.
            self._updater = Updater(self)
            self._updater.setHardwareID("none")
        else:
            self._updater.timer.reset()
        detector_rate = 1.0 / self._detector_period_sec
        self._freq_bounds = {
            "detector": {"min": 0.5 * detector_rate, "max": detector_rate},
            "mapping": {"min": 0.0},
            "publish": {"min": 0.0, "max": 1.0 / self._publish_period_sec},
        }
        self._freq = {name: FrequencyStatus(FrequencyStatusParam(bounds, 0.1, 5), name=f"{name} rate")
                      for name, bounds in self._freq_bounds.items()}
        for task in self._freq.values():
            self._updater.add(task)
        self._updater.add("mapping state", self._diagnose_state)
        self._diagnostic_task_names = [task.name for task in self._freq.values()] + ["mapping state"]
        self._last_diagnosed = {}

    def _diagnose_state(self, stat):
        with self._stats_lock:
            counters = dict(self._counters)
            failures = dict(self._failures_by_stage)
        new = {k: v - self._last_diagnosed.get(k, 0) for k, v in counters.items()}
        self._last_diagnosed = counters
        if new["failures"]:
            stat.summary(DiagnosticStatus.ERROR, f"{new['failures']} processing failures since last report")
        elif new["tf_failures"] or new["invalid_input"] or new["detector_timeouts"]:
            stat.summary(DiagnosticStatus.WARN, "frames dropped since last report")
        elif not self._active:
            stat.summary(DiagnosticStatus.WARN, "inactive")
        else:
            stat.summary(DiagnosticStatus.OK, "mapping")
        for key, value in counters.items():
            stat.add(key, str(value))
        for stage, value in failures.items():
            stat.add(f"failures.{stage}", str(value))
        stat.add("pending_frames", str(len(self._pending_frames)))
        stat.add("frames_waiting_for_tf", str(len(self._tf_waiting)))
        stat.add("detector_busy", str(self._detector_in_flight is not None))
        stat.add("grounding_outstanding", str(self._grounding_outstanding))
        pipeline = getattr(self, "pipeline", None)
        stat.add("instances", str(len(pipeline.object_map.objects) if pipeline is not None else 0))
        return stat

    def _tick(self, name: str) -> None:
        freq = getattr(self, "_freq", None)
        if freq:
            freq[name].tick()

    def _count(self, key: str) -> None:
        with self._stats_lock:
            self._counters[key] += 1

    def _count_failure(self, stage: str, exc: BaseException) -> None:
        with self._stats_lock:
            self._counters["failures"] += 1
            self._failures_by_stage[stage] = self._failures_by_stage.get(stage, 0) + 1
        self.get_logger().error(f"{stage} failed: {type(exc).__name__}: {exc}", throttle_duration_sec=5.0)

    @staticmethod
    def _empty_stats() -> dict:
        return {"frames": 0, "detections": 0, "publishes": 0, "stage_seconds": {},
                "detector_seconds": 0.0, "publish_seconds": 0.0}

    # ----------------------------------------------------------- input state
    def _reset_input_state(self) -> None:
        """Forget queued input and rate schedules (map load, time jumps). Caller holds the lock."""
        self._pending_frames.clear()
        self._pending_by_id.clear()  # in-flight results are discarded by frame ID
        self._tf_waiting.clear()
        self._scan_history.clear()
        self._last_input_stamp = self._last_detector_stamp = self._last_publish_stamp = -float("inf")

    def _restart_clock_epoch(self) -> None:
        """Time went backwards: forget queued input and let the pipeline accept the new
        epoch, rebasing the map's stamps the same way it does after a map load. Caller holds the lock."""
        self._reset_input_state()
        self.pipeline.begin_new_epoch()

    def _on_time_jump(self, jump) -> None:
        """Clock moved backwards (e.g. ``ros2 bag play --loop``): restart input bookkeeping (C8)."""
        with self._lock:
            if not self._configured:
                return
            self._restart_clock_epoch()
        self._count("time_jumps")
        self.get_logger().warning("clock jumped backwards; input queues and rate schedules reset")

    # ----------------------------------------------------------- persistence
    def _save_map(self, path: str) -> tuple[FilePath, int]:
        """Snapshot the map under the lock, then write it without holding the lock,
        so a save never stalls sensor processing."""
        with self._lock:
            snapshot = copy.copy(self.pipeline)
            snapshot.object_map = copy.deepcopy(self.pipeline.object_map)
        saved = snapshot.save(path, metadata={"world_frame": self.world_frame})
        return FilePath(saved), len(snapshot.object_map.objects)

    def _load_map(self, path: str) -> str:
        with self._lock:
            header = self.pipeline.load(path)
            self._last_result = None  # the next frame re-derives the graph from the restored map
            self._reset_input_state()
            self._published_marker_ids = set()
            self._marker_reset_pending = True
            self._cloud_signature = None
            if self._active:
                self._publish_marker_reset()
        return f"restored {header['num_instances']} instances from {path}"

    def _publish_marker_reset(self) -> None:
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.world_frame)
        self.obj_boxes_pub.publish(MarkerArray(markers=[Marker(header=header, action=Marker.DELETEALL)]))
        self.objects_pub.publish(Detection3DArray(header=header))
        self._marker_reset_pending = False

    def _on_save_map(self, request: SaveMap.Request, response: SaveMap.Response) -> SaveMap.Response:
        path = request.path.strip() or self.map_save_path
        if not path:
            response.success, response.message = False, "no path given and map_save_path is empty"
            return response
        try:
            saved, count = self._save_map(path)
            response.success, response.path = True, str(saved)
            response.message = f"saved {count} instances to {saved}"
            self.get_logger().info(response.message)
        except Exception as exc:  # noqa: BLE001 - report to the caller instead of killing the executor
            response.success, response.message = False, f"save failed: {exc}"
            self.get_logger().error(response.message)
        return response

    def _on_load_map(self, request: LoadMap.Request, response: LoadMap.Response) -> LoadMap.Response:
        path = request.path.strip() or self._param_str("map_load_path") or self.map_save_path
        if not path:
            response.success, response.message = False, "no path given and neither map_load_path nor map_save_path is set"
            return response
        try:
            response.success, response.message, response.path = True, self._load_map(path), path
            self.get_logger().info(response.message)
        except Exception as exc:  # noqa: BLE001
            response.success, response.message = False, f"load failed: {exc}"
            self.get_logger().error(response.message)
        return response

    def _autosave(self) -> None:
        try:
            saved, count = self._save_map(self.map_save_path)
            self.get_logger().debug(f"autosaved {count} instances to {saved}")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"autosave failed: {exc}")

    def _log_runtime_stats(self) -> None:
        """Report module rates in the terms of Sec. V-H: 2D segmentation (detector),
        3D mapping (geometric update at the sensor rate), and 4D scene graph output."""
        now = time.monotonic()
        with self._stats_lock:
            elapsed = max(now - self._stats_since, 1e-6)
            stats, self._stats = self._stats, self._empty_stats()
            self._stats_since = now
        frames, detections, publishes = (stats[k] for k in ("frames", "detections", "publishes"))
        if frames == 0 and detections == 0:
            return
        stages = " ".join(f"{k}={1e3 * v / max(frames, 1):.1f}ms" for k, v in stats["stage_seconds"].items())
        objects = len(self.pipeline.object_map.objects)
        self.get_logger().info(
            f"runtime: detector {detections / elapsed:.2f} Hz, 3D mapping {frames / elapsed:.2f} Hz, "
            f"scene graph published {publishes / elapsed:.2f} Hz, {objects} instances | per frame: {stages} | "
            f"per detection: {1e3 * stats['detector_seconds'] / max(detections, 1):.1f}ms; "
            f"per publish: {1e3 * stats['publish_seconds'] / max(publishes, 1):.1f}ms"
        )

    # ------------------------------------------------------------- grounding
    def _robot_position(self) -> np.ndarray | None:
        """Latest robot position in the world frame (C7: newest transform, not ``now()``)."""
        try:
            return self._lookup_se3(self.world_frame, self.robot_frame, rclpy.time.Time().to_msg())[:3, 3]
        except tf2_ros.TransformException as exc:
            self.get_logger().warning(
                f"robot pose {self.world_frame} -> {self.robot_frame} unavailable ({exc}); "
                "grounding uses the whole graph and a fixed goal height", throttle_duration_sec=10.0)
            return None

    def _reserve_grounding_slot(self) -> bool:
        with self._slot_lock:
            if self._grounding_outstanding >= self._grounding_max_queue:
                return False
            self._grounding_outstanding += 1
            return True

    def _release_grounding_slot(self) -> None:
        with self._slot_lock:
            self._grounding_outstanding = max(self._grounding_outstanding - 1, 0)

    def _prepare_grounding(self, instruction: str, request_id: str, local_radius_m: float | None = None,
                           feedback=None, on_done=None) -> _GroundingJob:
        """Snapshot and serialize the graph under the lock (never races the mapping callbacks)."""
        robot = self._robot_position()
        with self._lock:
            if not self._active:
                raise _GroundingUnavailable("node is not active")
            result = self._last_result
            if result is None:
                raise _GroundingUnavailable("no map update yet")
            configured_radius = self.grounder.local_radius_m
            if local_radius_m is not None and local_radius_m > 0:
                self.grounder.local_radius_m = float(local_radius_m)
            try:
                request = self.grounder.prepare(instruction, result.objects, result.scene_graph, robot,
                                                now=result.stamp)
            finally:
                self.grounder.local_radius_m = configured_radius
            bboxes = {o.instance_id: np.asarray(o.bbox3d, dtype=np.float64).copy() for o in result.objects}
            labels = {o.instance_id: o.label for o in result.objects}
            obstacles = {o.instance_id: bboxes[o.instance_id] for o in result.objects
                         if o.status != ObjectStatus.DISAPPEARED}
        goal_z = float(robot[2]) if robot is not None and self._goal_use_robot_z else self._goal_z_m
        return _GroundingJob(request_id=request_id, instruction=instruction, request=request, bboxes=bboxes,
                             labels=labels, obstacles=obstacles,
                             robot_xy=None if robot is None else np.asarray(robot[:2], dtype=np.float64),
                             goal_z=goal_z, feedback=feedback, on_done=on_done)

    def _on_query(self, msg: String) -> None:
        """Legacy interface: instruction (plain text, or JSON with ``instruction`` and
        ``request_id``) in -> JSON answer, goal and waypoints out."""
        text = msg.data.strip()
        request_id = None
        if text.startswith("{"):
            try:
                payload = json.loads(text)
                text = str(payload.get("instruction", "")).strip()
                request_id = payload.get("request_id")
            except (ValueError, AttributeError):
                pass
        if not text:
            return
        with self._slot_lock:
            self._query_counter += 1
            request_id = str(request_id) if request_id is not None else f"query-{self._query_counter}"
        if not self._reserve_grounding_slot():
            self._publish_answer_error(text, request_id, "grounding queue is full; retry later")
            return
        try:
            job = self._prepare_grounding(text, request_id, on_done=self._publish_topic_answer)
        except _GroundingUnavailable as exc:
            self._release_grounding_slot()
            self.get_logger().warning(f"query '{text}' ignored: {exc}")
            self._publish_answer_error(text, request_id, str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self._release_grounding_slot()
            self._count_failure("grounding", exc)
            self._publish_answer_error(text, request_id, f"{type(exc).__name__}: {exc}")
            return
        self._grounding_jobs.put(job)

    def _publish_answer_error(self, instruction: str, request_id: str, error: str) -> None:
        self.answer_pub.publish(String(data=json.dumps({
            "request_id": request_id, "instruction": instruction, "target_ids": [], "waypoints": [],
            "goals": [], "unresolved_ids": [], "response": "", "error": error})))

    def _publish_topic_answer(self, job: _GroundingJob) -> None:
        outcome = job.outcome
        payload = outcome.result.to_dict() if outcome.result is not None else {
            "instruction": job.instruction, "target_ids": [], "waypoints": [], "unresolved_ids": [],
            "response": outcome.response, "error": outcome.message}
        payload["request_id"] = job.request_id
        payload["goals"] = [list(goal) for goal in outcome.goals]
        self.answer_pub.publish(String(data=json.dumps(payload)))

    def _grounding_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._grounding_jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:
                return
            try:
                if job.cancelled:
                    continue
                self._send_feedback(job, "querying")
                result = self.grounder.complete(job.request)
                if self._stop_event.is_set():
                    return
                self._send_feedback(job, "parsing")
                job.outcome = self._finish_grounding(job, result)
            except Exception as exc:  # noqa: BLE001 - a VLM failure must not kill the worker
                job.outcome = _GroundingOutcome(False, f"grounding failed: {type(exc).__name__}: {exc}")
                self._count_failure("grounding", exc)
            finally:
                self._release_grounding_slot()
                job.done.set()
            if job.cancelled or job.outcome is None:
                continue
            try:
                if job.on_done is not None:
                    job.on_done(job)
                self._publish_goals(job.outcome)
            except Exception as exc:  # noqa: BLE001
                self._count_failure("grounding output", exc)

    @staticmethod
    def _send_feedback(job: _GroundingJob, stage: str) -> None:
        if job.feedback is not None and not job.cancelled:
            try:
                job.feedback(stage)
            except Exception:  # noqa: BLE001 - the goal may have been cancelled meanwhile
                pass

    def _finish_grounding(self, job: _GroundingJob, result: GroundingResult) -> _GroundingOutcome:
        if result.error:
            self.get_logger().warning(f"grounding '{job.instruction}': {result.error}")
        else:
            self.get_logger().info(f"grounding '{job.instruction}' -> instances {result.target_ids}")
        goals, labels = [], []
        for target in result.target_ids:
            bbox = job.bboxes.get(target)
            if bbox is None:
                continue
            others = [b for i, b in job.obstacles.items() if i != target]
            x, y, yaw = approach_pose(bbox, job.robot_xy, self._goal_standoff_m, others, self._goal_clearance_m)
            goals.append((x, y, job.goal_z, yaw))
            labels.append(job.labels.get(target, ""))
        ok = result.error is None and bool(goals)
        message = result.error or ("" if goals else "no answered instance is in the map")
        return _GroundingOutcome(ok, message or f"{len(goals)} goals",
                                 target_ids=[i for i in result.target_ids if i in job.bboxes],
                                 target_labels=labels, unresolved_ids=list(result.unresolved_ids),
                                 goals=goals, response=result.response, result=result)

    def _goal_poses(self, outcome: _GroundingOutcome) -> list[PoseStamped]:
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.world_frame)
        poses = []
        for x, y, z, yaw in outcome.goals:
            pose = PoseStamped(header=header)
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = x, y, z
            q = _yaw_quaternion(yaw)
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = q
            poses.append(pose)
        return poses

    def _publish_goals(self, outcome: _GroundingOutcome) -> None:
        poses = self._goal_poses(outcome)
        if not poses:
            return
        self.waypoints_pub.publish(Path(header=poses[0].header, poses=poses))
        self.goal_pub.publish(poses[0])
        if self._nav2_send_goal:
            self._send_nav2_goal(poses[0])

    def _send_nav2_goal(self, pose: PoseStamped) -> None:
        """Forward an approach pose to Nav2. nav2_msgs is imported lazily, so the
        node runs without Nav2 installed while this is disabled."""
        try:
            from nav2_msgs.action import NavigateToPose
            from rclpy.action import ActionClient
        except ImportError:
            self.get_logger().error("nav2_send_goal is set but nav2_msgs is not installed", once=True)
            return
        if self._nav2_client is None:
            self._nav2_client = ActionClient(self, NavigateToPose, self._param_str("nav2_action_name"),
                                             callback_group=self._query_group)
        if not self._nav2_client.server_is_ready():
            self.get_logger().warning("Nav2 NavigateToPose server not available; goal not sent",
                                      throttle_duration_sec=10.0)
            return
        future = self._nav2_client.send_goal_async(NavigateToPose.Goal(pose=pose))
        future.add_done_callback(lambda f: self.get_logger().info(
            f"Nav2 goal {'accepted' if f.result() and f.result().accepted else 'rejected'}"))

    # ------------------------------------------------------ grounding action
    def _on_ground_goal(self, goal_request) -> GoalResponse:
        if not self._active or not goal_request.instruction.strip():
            return GoalResponse.REJECT
        if not self._reserve_grounding_slot():
            self.get_logger().warning("grounding queue full; goal rejected", throttle_duration_sec=5.0)
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _execute_ground_instruction(self, goal_handle):
        goal = goal_handle.request
        result = GroundInstruction.Result()

        def feedback(stage: str) -> None:
            goal_handle.publish_feedback(GroundInstruction.Feedback(stage=stage))

        job = None
        try:
            feedback("serializing")
            job = self._prepare_grounding(goal.instruction.strip(), uuid.UUID(bytes=bytes(goal_handle.goal_id.uuid)).hex,
                                          local_radius_m=goal.local_radius_m, feedback=feedback)
        except Exception as exc:  # noqa: BLE001 - _GroundingUnavailable or a serialization error
            self._release_grounding_slot()
            if not isinstance(exc, _GroundingUnavailable):
                self._count_failure("grounding", exc)
            result.success, result.message = False, str(exc)
            goal_handle.abort()
            return result
        self._grounding_jobs.put(job)
        feedback("queued")
        while not job.done.wait(0.05):
            if goal_handle.is_cancel_requested:
                job.cancelled = True  # a running model call finishes in the background and is discarded
                goal_handle.canceled()
                result.message = "canceled"
                return result
            if self._stop_event.is_set():
                job.cancelled = True
                goal_handle.abort()
                result.message = "node shutting down"
                return result
        outcome = job.outcome or _GroundingOutcome(False, "grounding did not complete")
        poses = self._goal_poses(outcome)
        result.success, result.message = outcome.success, outcome.message
        result.target_ids = [int(i) for i in outcome.target_ids if int(i) >= 0]
        result.target_labels = list(outcome.target_labels)
        result.unresolved_ids = [int(i) for i in outcome.unresolved_ids if int(i) >= 0]
        result.goals = poses
        result.path = Path(header=poses[0].header if poses else Header(frame_id=self.world_frame), poses=poses)
        result.response = outcome.response
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        elif outcome.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    # ---------------------------------------------------------------- callback
    def _on_synced_frame(self, rgb_msg, info_msg: CameraInfo, depth_msg, odom_msg: Odometry | None = None) -> None:
        # odom_msg's own pose fields are not read directly: a well-behaved
        # SLAM backbone also broadcasts the same pose as a dynamic TF
        # transform, so resolving world_frame -> camera_frame (and
        # <point cloud frame> -> camera_frame) through TF2 handles the full
        # chain -- dynamic odometry composed with whatever static camera
        # extrinsic is in the TF tree -- without this node hardcoding either.
        # By default, Odometry paces processing at the backbone's pose-update
        # rate. With sync_odometry=false, only RGB/CameraInfo/depth are synced;
        # TF is still required (including for an explicitly stationary camera).
        with self._lock:
            if not self._active:
                return
            try:
                self._drain_detection_results_locked()
                stamp = _stamp_to_seconds(rgb_msg.header.stamp)
                latest = max(self._last_input_stamp, self._tf_waiting[-1].stamp if self._tf_waiting else -math.inf)
                if stamp <= latest:
                    if 0 < self._jump_reset_sec < latest - stamp:
                        self.get_logger().warning("input stamps jumped backwards; input state reset")
                        self._count("time_jumps")
                        self._restart_clock_epoch()
                    else:
                        self._count("out_of_order")
                        self.get_logger().warning("skipping duplicate or out-of-order sensor frame",
                                                  throttle_duration_sec=5.0)
                        return
                self._tf_waiting.append(_WaitingFrame(rgb_msg, info_msg, depth_msg, stamp, time.monotonic()))
                self._admit_waiting_frames()
                self._flush_pending_frames()
            except Exception as exc:  # noqa: BLE001 - never kill the executor (R11)
                self._count_failure("frame input", exc)

    def _admit_waiting_frames(self) -> None:
        """Turn frames whose transforms are available into observations, in arrival order.

        A frame whose TF has not arrived yet waits up to ``tf_wait_sec`` (and
        holds later frames behind it) instead of being dropped immediately.
        """
        while self._tf_waiting:
            waiting = self._tf_waiting[0]
            try:
                T_world_from_cam = self._lookup_se3(self.world_frame, self.camera_frame, waiting.rgb.header.stamp)
                T_world_from_cloud = None if self.depth_from_image else self._lookup_se3(
                    self.world_frame, waiting.depth.header.frame_id, waiting.depth.header.stamp)
            except tf2_ros.TransformException as exc:
                if (time.monotonic() - waiting.arrived < self._tf_wait_sec
                        and len(self._tf_waiting) <= self._max_pending_frames):
                    return
                self._tf_waiting.popleft()
                self._count("tf_failures")
                self.get_logger().warning(f"TF lookup failed, skipping frame: {exc}", throttle_duration_sec=5.0)
                continue
            self._tf_waiting.popleft()
            self._admit_frame(waiting, T_world_from_cam, T_world_from_cloud)

    def _camera_intrinsics(self, info_msg: CameraInfo) -> tuple[CameraIntrinsics, CameraInfo]:
        """Intrinsics for projecting into the received image (C29).

        Rectified input projects with P (its left 3x3); D and R then describe
        the raw sensor and are irrelevant. Raw input must be undistorted
        (D == 0), since the pinhole model here does not model distortion.
        Returns the intrinsics and the calibration that matches the image
        as delivered (D = 0, R = I, K = P for rectified input).
        """
        if self._images_rectified:
            try:
                intrinsics = camera_info_to_intrinsics(info_msg, rectified=True)
            except ValueError as exc:
                raise ValueError(f"camera_images_rectified=true needs a valid projection matrix P: {exc}") from exc
            return intrinsics, rectified_camera_info(info_msg)
        if camera_info_has_distortion(info_msg):
            raise ValueError("CameraInfo has non-zero distortion D; rectify the images (image_proc) and set "
                             "camera_images_rectified:=true")
        return camera_info_to_intrinsics(info_msg, allow_distortion=False), info_msg

    def _admit_frame(self, waiting: _WaitingFrame, T_world_from_cam: np.ndarray,
                     T_world_from_cloud: np.ndarray | None) -> None:
        rgb_msg, info_msg, depth_msg, stamp = waiting.rgb, waiting.info, waiting.depth, waiting.stamp
        detection_due = (self._detector_in_flight is None
                         and _rate_due(self._last_detector_stamp, stamp, self._detector_period_sec))
        try:
            intrinsics, calibration = self._camera_intrinsics(info_msg)
            if self.dense_annotations_pub is not None and (
                    rgb_msg.header.frame_id != self.camera_frame or info_msg.header.frame_id != self.camera_frame):
                raise ValueError("dense annotation export needs images in camera_frame")
            # Decode RGB only for the frames that reach the detector (and so
            # the annotated image and dense export); others need only its size.
            if detection_due:
                rgb = self._decode_rgb(rgb_msg, intrinsics)
            else:
                self._check_image_size(rgb_msg, intrinsics)
                rgb = None
        except ValueError as exc:
            self._count("invalid_input")
            self.get_logger().error(f"invalid camera input: {exc}", throttle_duration_sec=5.0)
            return

        if self.depth_from_image:
            # Depth already aligned to the RGB camera (e.g. an RGB-D driver's aligned stream).
            depth = depth_image_to_meters(depth_msg, self.depth_scale)
            if depth.shape != (intrinsics.height, intrinsics.width):
                self._count("invalid_input")
                self.get_logger().error(
                    f"depth image {depth.shape[1]}x{depth.shape[0]} does not match CameraInfo "
                    f"{intrinsics.width}x{intrinsics.height}; use the driver's depth-aligned-to-color stream",
                    throttle_duration_sec=5.0)
                return
        else:
            # Rasterize the point cloud into the camera to obtain the raw
            # sensor depth D(u) needed by geometric consistency. The cloud is
            # placed in the world at its own stamp; every scan is projected
            # through the camera at the RGB stamp.
            points_cloud_frame = pointcloud_to_xyz(depth_msg)
            T_cam_from_world = invert_se3(T_world_from_cam)
            if self._scan_history.maxlen > 1:
                # A single sparse LiDAR scan covers few pixels; the last N
                # scans, carried in the world frame, give a denser depth image.
                self._scan_history.append(transform_points(T_world_from_cloud, points_cloud_frame))
                points_cam = transform_points(T_cam_from_world, self._scan_history.points())
            else:
                points_cam = transform_points(T_cam_from_world @ T_world_from_cloud, points_cloud_frame)
            depth = rasterize_depth(points_cam, intrinsics.K, intrinsics.width, intrinsics.height)

        observation = Observation(
            stamp=stamp,
            pose=StampedPose(stamp=stamp, T_world_from_frame=T_world_from_cam),
            intrinsics=intrinsics,
            frame_id=self._next_frame_id,
            rgb=rgb,
            depth=depth,
            detections=[],
            detections_evaluated=False,
        )
        self._next_frame_id += 1
        self._last_input_stamp = stamp
        pending = _PendingFrame(observation, Header(stamp=rgb_msg.header.stamp, frame_id=rgb_msg.header.frame_id),
                                camera_info=calibration if self.dense_annotations_pub is not None else None)
        self._pending_frames.append(pending)
        self._pending_by_id[observation.frame_id] = pending

        if detection_due:
            pending.detector_started_at = time.monotonic()
            self._detector_in_flight = observation.frame_id
            self._detection_jobs.put(observation)
            self._last_detector_stamp = _advance_schedule(
                self._last_detector_stamp, stamp, self._detector_period_sec)

    def _detector_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                observation = self._detection_jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            evaluated = True
            started_at = time.monotonic()
            try:
                detections = self.detector.detect(
                    observation.rgb,
                    prompts=self.prompts,
                    frame_id=observation.frame_id,
                )
                for detection in detections:
                    detection.validate_mask((observation.intrinsics.height, observation.intrinsics.width))
            except Exception as exc:  # noqa: BLE001 - a detector failure must not kill the mapping loop
                if self._stop_event.is_set():
                    return
                self._count_failure("detector", exc)
                detections = []
                evaluated = False
            if self._stop_event.is_set():
                return
            with self._stats_lock:
                self._stats["detections"] += 1
                self._stats["detector_seconds"] += time.monotonic() - started_at
            self._tick("detector")
            self._detection_results.put((observation.frame_id, detections, time.monotonic(), evaluated))
            # The guard condition is destroyed under the same lock during
            # teardown, so it is never triggered after destruction (C38).
            with self._guard_lock:
                if self._detection_ready is not None and not self._stop_event.is_set():
                    self._detection_ready.trigger()

    def _drain_detection_results(self) -> None:
        with self._lock:
            if not self._configured:
                return
            try:
                self._drain_detection_results_locked()
                self._admit_waiting_frames()
                self._flush_pending_frames()
            except Exception as exc:  # noqa: BLE001
                self._count_failure("result drain", exc)

    def _drain_detection_results_locked(self) -> None:
        while True:
            try:
                frame_id, detections, finished_at, evaluated = self._detection_results.get_nowait()
            except queue.Empty:
                break
            if self._detector_in_flight == frame_id:
                self._detector_in_flight = None
            pending = self._pending_by_id.get(frame_id)
            if pending is None or pending.detector_started_at is None:
                continue  # timed out, forced through by the buffer limit, or invalidated by load/time jump
            if finished_at - pending.detector_started_at <= self._detector_timeout_sec:
                pending.observation.detections = detections
                pending.observation.detections_evaluated = evaluated
                pending.annotate = True
                pending.detector_started_at = None
            # An overdue result leaves the frame waiting, so the flush below
            # expires it and reports the timeout consistently.

    def _flush_pending_frames(self) -> None:
        while self._pending_frames:
            pending = self._pending_frames[0]
            if pending.detector_started_at is not None:
                timed_out = time.monotonic() - pending.detector_started_at >= self._detector_timeout_sec
                full = len(self._pending_frames) > self._max_pending_frames
                if not timed_out and not full:
                    break
                self._count("detector_timeouts")
                self.get_logger().warning(
                    "detector deadline or frame buffer limit reached; fusing without detections",
                    throttle_duration_sec=5.0)
            self._pending_frames.popleft()
            self._pending_by_id.pop(pending.observation.frame_id)
            if pending.annotate and pending.observation.detections_evaluated:
                try:
                    self._publish_dense_camera_annotations(pending)
                except Exception as exc:  # noqa: BLE001
                    self._count_failure("dense annotations", exc)
            try:
                result = self._process_and_publish(pending.observation, pending.header)
            except Exception as exc:  # noqa: BLE001 - drop this frame, keep mapping (R11)
                self._count_failure("mapping", exc)
                continue
            if pending.annotate:
                try:
                    self._publish_annotated_image(pending.observation, result, pending.header)
                except Exception as exc:  # noqa: BLE001
                    self._count_failure("annotated image", exc)

    def _process_and_publish(self, observation: Observation, header: Header) -> FrameResult:
        result = self.pipeline.process_frame(observation)
        self._last_result = result
        self._tick("mapping")
        with self._stats_lock:
            self._stats["frames"] += 1
            for stage, seconds in result.timings.items():
                self._stats["stage_seconds"][stage] = self._stats["stage_seconds"].get(stage, 0.0) + seconds
        if _rate_due(self._last_publish_stamp, observation.stamp, self._publish_period_sec):
            self._last_publish_stamp = _advance_schedule(
                self._last_publish_stamp, observation.stamp, self._publish_period_sec)
            started_at = time.monotonic()
            try:
                self._publish_result(result, header)
            except Exception as exc:  # noqa: BLE001 - the map update itself succeeded
                self._count_failure("publish", exc)
                return result
            self._tick("publish")
            with self._stats_lock:
                self._stats["publishes"] += 1
                self._stats["publish_seconds"] += time.monotonic() - started_at
        return result

    def _lookup_se3(self, target_frame: str, source_frame: str, stamp) -> np.ndarray:
        """4x4 SE(3) transform mapping ``source_frame``-expressed points/poses
        into ``target_frame`` coordinates, per REP 105 TF2 lookup semantics.
        """
        return transform_to_se3(self.tf_buffer.lookup_transform(
            target_frame, source_frame, rclpy.time.Time.from_msg(stamp)))

    @staticmethod
    def _check_image_size(rgb_msg, intrinsics: CameraIntrinsics) -> None:
        """Size check without decoding (raw images only; compressed ones are checked when decoded)."""
        if isinstance(rgb_msg, Image) and (rgb_msg.height, rgb_msg.width) != (intrinsics.height, intrinsics.width):
            raise ValueError(
                f"image {rgb_msg.width}x{rgb_msg.height} does not match CameraInfo after ROI/binning "
                f"({intrinsics.width}x{intrinsics.height}); supply matching image calibration")

    def _decode_rgb(self, rgb_msg, intrinsics: CameraIntrinsics) -> np.ndarray:
        """Decode at the ROI/binning-adjusted CameraInfo size without stretching pixels."""
        rgb = image_to_numpy(rgb_msg)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[:, :, None], 3, axis=2)
        if rgb.shape[:2] != (intrinsics.height, intrinsics.width):
            raise ValueError(
                f"image {rgb.shape[1]}x{rgb.shape[0]} does not match CameraInfo after ROI/binning "
                f"({intrinsics.width}x{intrinsics.height}); supply matching image calibration")
        return np.ascontiguousarray(rgb)

    # ---------------------------------------------------------------- publish
    def _publish_dense_camera_annotations(self, pending: _PendingFrame) -> None:
        if self.dense_annotations_pub is None:
            return
        from semantic_mapping.dense_cloud_io import annotations_to_msg
        try:
            msg = annotations_to_msg(pending.header, pending.camera_info, pending.observation.detections,
                                     self._dense_annotation_model, source="yoloe")
        except ValueError as exc:
            self.get_logger().warning(f"Dense camera masks skipped; object tracking continues: {exc}",
                                      throttle_duration_sec=5.0)
            return
        self.dense_annotations_pub.publish(msg)

    def _publish_annotated_image(self, observation: Observation, result: FrameResult, header: Header) -> None:
        if observation.rgb is None or self.annotated_image_pub.get_subscription_count() == 0:
            return
        try:
            import cv2
        except ImportError:
            self.get_logger().warning("python3-opencv not installed; annotated image disabled", once=True)
            return

        image = np.ascontiguousarray(observation.rgb.copy())
        for detection, instance_id in zip(observation.detections, result.detection_instance_ids):
            r, g, b = (int(c * 255) for c in _label_color(detection.label))
            x1, y1, x2, y2 = (int(round(v)) for v in detection.bbox)
            if detection.mask is not None:
                pixels = np.flatnonzero(detection.mask)  # index the mask once, then work on a flat view
                flat = image.reshape(-1, 3)
                flat[pixels] = (0.6 * flat[pixels] + 0.4 * np.array([r, g, b])).astype(np.uint8)
            cv2.rectangle(image, (x1, y1), (x2, y2), (r, g, b), 2)
            tag = f"#{instance_id} {detection.label} {detection.score:.2f}" if instance_id >= 0 \
                else f"{detection.label} {detection.score:.2f} (dropped)"
            cv2.putText(image, tag, (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (r, g, b), 1, cv2.LINE_AA)

        self.annotated_image_pub.publish(numpy_to_image(image, "rgb8", header))

    def _displayed_objects(self, result: FrameResult) -> list:
        """Scene-graph nodes that exist in the map, minus retired ones unless history is shown."""
        by_id = {obj.instance_id: obj for obj in result.objects}
        node_ids = set(result.scene_graph.node_ids) if result.scene_graph else set()
        node_ids.intersection_update(by_id)
        return [by_id[i] for i in sorted(node_ids)
                if self._include_history or by_id[i].status != ObjectStatus.DISAPPEARED]

    def _publish_result(self, result: FrameResult, header: Header) -> None:
        header = Header(stamp=header.stamp, frame_id=self.world_frame)
        displayed = self._displayed_objects(result)
        self._publish_object_points(result, header)
        self._publish_object_boxes(result, header, displayed)
        self._publish_objects(header, displayed)
        # JSON consumers use semantic_mapping.serialization on demand. Do not
        # serialize the entire history here only to discard it at sensor rate.

    def _publish_object_points(self, result: FrameResult, header: Header) -> None:
        # One structured array for the whole map, serialized by create_cloud in
        # a single copy: a Python row per point took 140 ms for a room-sized
        # map and over a second for a building (doc/audit-2026-09.md, P1).
        objects = [obj for obj in result.objects
                   if self._include_history or obj.status != ObjectStatus.DISAPPEARED]
        counts = [obj.points_world.shape[0] for obj in objects]
        # The cloud is latched; republish it only when its content changed (P6).
        signature = tuple((obj.instance_id, obj.label, n, float(obj.points_world.sum()) if n else 0.0)
                          for obj, n in zip(objects, counts))
        if signature == self._cloud_signature:
            return
        cloud = np.zeros(int(sum(counts)), dtype=_POINT_DTYPE)
        offset = 0
        for obj, n in zip(objects, counts):
            if n == 0:
                continue
            block = cloud[offset:offset + n]
            block["x"] = obj.points_world[:, 0]
            block["y"] = obj.points_world[:, 1]
            block["z"] = obj.points_world[:, 2]
            block["rgb"] = _packed_rgb_float(obj.label)
            offset += n
        self.obj_points_pub.publish(pc2.create_cloud(header, _POINT_FIELDS, cloud))
        self._cloud_signature = signature

    def _publish_object_boxes(self, result: FrameResult, header: Header, displayed: list | None = None) -> None:
        now = result.stamp or _stamp_to_seconds(header.stamp)
        if displayed is None:
            displayed = self._displayed_objects(result)
        node_ids = {obj.instance_id for obj in displayed}
        marker_array = MarkerArray()
        if self._marker_reset_pending:
            # Clear boxes left in RViz by an earlier run or the map before a load (R17).
            marker_array.markers.append(Marker(header=header, action=Marker.DELETEALL))
            self._marker_reset_pending = False
            self._published_marker_ids = set()

        for instance_id in sorted(self._published_marker_ids - node_ids):
            for namespace in ("obj_boxes", "obj_labels"):
                marker_array.markers.append(Marker(header=header, ns=namespace, id=instance_id, action=Marker.DELETE))

        for obj in displayed:
            r, g, b = _label_color(obj.label)
            xmin, ymin, zmin, xmax, ymax, zmax = obj.bbox3d

            box = Marker()
            box.header = header
            box.ns = "obj_boxes"
            box.id = obj.instance_id
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose.position.x = float((xmin + xmax) / 2.0)
            box.pose.position.y = float((ymin + ymax) / 2.0)
            box.pose.position.z = float((zmin + zmax) / 2.0)
            box.pose.orientation.w = 1.0
            box.scale.x = float(max(xmax - xmin, 1e-3))
            box.scale.y = float(max(ymax - ymin, 1e-3))
            box.scale.z = float(max(zmax - zmin, 1e-3))
            box.color = ColorRGBA(r=r, g=g, b=b, a=0.35)
            marker_array.markers.append(box)

            label_marker = Marker()
            label_marker.header = header
            label_marker.ns = "obj_labels"
            label_marker.id = obj.instance_id
            label_marker.type = Marker.TEXT_VIEW_FACING
            label_marker.action = Marker.ADD
            label_marker.pose.position.x = box.pose.position.x
            label_marker.pose.position.y = box.pose.position.y
            label_marker.pose.position.z = float(zmax + 0.1)
            label_marker.scale.z = 0.15
            label_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            status = obj.status.value
            if obj.status == ObjectStatus.OCCLUDED:
                measured = obj.geometry_stamp if obj.geometry_stamp is not None else obj.latest_stamp
                status = f"occluded {max(now - measured, 0.0):.0f}s"
            label_marker.text = f"{obj.instance_id}:{obj.label} ({status})"
            marker_array.markers.append(label_marker)

        self.obj_boxes_pub.publish(marker_array)
        self._published_marker_ids = node_ids

    def _publish_objects(self, header: Header, displayed: list) -> None:
        """Typed map output (R12): one Detection3D per object, its box in the world frame."""
        array = Detection3DArray(header=header)
        for obj in displayed:
            xmin, ymin, zmin, xmax, ymax, zmax = (float(v) for v in obj.bbox3d)
            detection = Detection3D(header=header, id=str(obj.instance_id))
            center = detection.bbox.center
            center.position.x, center.position.y, center.position.z = \
                (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0
            center.orientation.w = 1.0
            detection.bbox.size.x = max(xmax - xmin, 0.0)
            detection.bbox.size.y = max(ymax - ymin, 0.0)
            detection.bbox.size.z = max(zmax - zmin, 0.0)
            beliefs = sorted(obj.label_belief.items(), key=lambda kv: kv[1], reverse=True)[:_MAX_HYPOTHESES] \
                or [(obj.label, 0.0)]
            for label, score in beliefs:
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(label)
                hypothesis.hypothesis.score = float(score)
                hypothesis.pose.pose = center
                detection.results.append(hypothesis)
            array.detections.append(detection)
        self.objects_pub.publish(array)


def main(args: list[str] | None = None) -> None:
    run_node(SemanticMappingNode, args, multithreaded=True)


if __name__ == "__main__":
    main()
