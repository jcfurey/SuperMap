"""ROS 2 full-cloud mapping, independent of camera availability and model inference.

A managed (lifecycle) node. Geometry comes from ``PointCloud2`` in any frame
with TF to ``world_frame``; optional per-image masks arrive as typed
``supermap_msgs/CameraAnnotations`` (from ``dense_yoloe_node`` or the object
node), or as manual JSON on a separate, disabled-by-default ``std_msgs/String``
topic. Outputs:

* ``output_cloud_topic``: the input cloud, byte-for-byte, plus label fields
  (published per processed cloud; stream QoS).
* ``voxel_map_topic``: world-frame voxel map (latched, throttled by
  ``publish_period_s``).
* ``regions_topic``: ``supermap_msgs/RegionArray`` (latched, throttled).
* ``/diagnostics``: rates, TF failures, drops and queue depths.

Callback groups: clouds (and the TF retry timer), annotations, CameraInfo and
publishing each get their own mutually exclusive group, so mask reception and
calibration stay live during long segmentation runs; run it on a
``MultiThreadedExecutor`` (``main`` does).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields, replace
import json
import math
import threading
import time
from typing import Any

import numpy as np
from diagnostic_msgs.msg import DiagnosticStatus
import diagnostic_updater
from rcl_interfaces.msg import SetParametersResult
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, PointCloud2
from std_msgs.msg import Header, String
from supermap_msgs.msg import CameraAnnotations, RegionArray
import tf2_ros

from semantic_mapping.dense_cloud import DenseCloudConfig, DenseCloudPipeline
from semantic_mapping.dense_cloud_io import (
    ANNOTATION_FIELDS, AnnotationLimits, annotate_pointcloud_message, camera_labels_from_dict,
    camera_labels_from_msg, check_rectified_camera_info, dumps_json, full_pointcloud_xyz,
    region_array_msg, result_metadata, voxel_map_message, _transform_msg_to_se3,
)
from semantic_mapping.platform_mask import exclude_platform
from semantic_mapping.ros_msgs import camera_info_to_intrinsics, stamp_to_seconds, transform_to_se3
from semantic_mapping.ros_node_utils import AutostartLifecycleNode, TransitionCallbackReturn, declare, run_node
from semantic_mapping.ros_platform_mask import PlatformMaskSource, PlatformUnavailable, declare_platform_mask_parameters

_THROTTLE = 5.0

# DenseCloudConfig fields: description and optional range.
_CONFIG_DOCS: dict[str, tuple[str, tuple[float, float] | None]] = {
    "input_mode": ("snapshot: each cloud is an authoritative map/submap; scan: union registered scans", None),
    "voxel_size": ("World voxel edge length (m)", (1e-3, 10.0)),
    "neighbor_radius": ("Surface-graph neighbour radius (m)", (1e-3, 10.0)),
    "normal_radius": ("Normal estimation radius (m)", (1e-3, 10.0)),
    "max_neighbors": ("Neighbours per voxel for normals and graph edges", (3, 256)),
    "normal_angle_deg": ("Max normal angle between connected voxels (deg)", (0.1, 89.9)),
    "plane_tolerance": ("Max point-to-plane offset between connected voxels (m)", (1e-4, 10.0)),
    "min_region_voxels": ("Smaller surface components stay unsegmented", (1, 1_000_000)),
    "max_map_voxels": ("Capacity; beyond it a cloud is rejected, never cropped", (1, 100_000_000)),
    "chunk_size": ("KD-tree query batch size", (1, 1_000_000)),
    "kdtree_workers": ("Threads for KD-tree neighbour queries; -1 = all cores (results do not depend on it)",
                       (-1, 1024)),
    "camera_depth_tolerance": ("Z-buffer / measured-depth agreement (m)", (1e-4, 10.0)),
    "max_camera_time_delta": ("Max |image stamp - cloud stamp| for labels (s)", (0.0, 60.0)),
    "min_camera_score": ("Ignore masks scored below this", (0.0, 1.0)),
    "label_propagation_radius": ("Propagate direct labels within a region up to this distance (m); 0 = off",
                                 (0.0, 100.0)),
    "label_ttl_sec": ("Forget labels older than this (s); 0 = keep", (0.0, 1e9)),
    "allow_yoloe_labels": ("Accept masks whose source is 'yoloe' (explicit policy exception)", None),
    "camera_splat_radius": ("Physical footprint radius of a projected point in the z-buffer (m); 0 = off",
                            (0.0, 10.0)),
    "camera_max_splat_pixels": ("Cap on a point's z-buffer footprint radius (px)", (1, 64)),
    "camera_splat_occlusion_gap": ("A footprint this far in front hides a point (m)", (0.0, 100.0)),
    "mask_depth_gap": ("Split a mask's points into depth layers at gaps above this (m); 0 = off", (0.0, 100.0)),
    "mask_depth_gap_ratio": ("Depth-relative layer gap (fraction of depth)", (0.0, 10.0)),
    "mask_min_layer_points": ("Nearest layer with at least this many points takes the label", (1, 100_000)),
    "max_labels": ("Maximum distinct semantic labels in one map", (1, 1_000_000)),
    "incremental_segmentation": ("Re-segment only near changed geometry", None),
    "ground_exclusion": ("Keep labels off the ground under an object whose mask bled onto it (world z up)", None),
    "ground_clearance": ("Points less than this above the fitted local ground are ground (m)", (1e-3, 10.0)),
    "ground_max_slope": ("Steepest local ground accepted (rise over run)", (1e-3, 10.0)),
    "ground_context_px": ("Margin around a mask whose points also inform the ground fit (px)", (0, 10_000)),
    "ground_surface_labels": ("Classes that are the ground itself; their masks keep ground points", None),
}

# Safe to change while running: read per annotation / per publish.
_RUNTIME_CONFIG = {"min_camera_score", "camera_depth_tolerance", "max_camera_time_delta",
                   "label_propagation_radius", "label_ttl_sec", "camera_splat_occlusion_gap",
                   "mask_depth_gap", "mask_depth_gap_ratio", "mask_min_layer_points",
                   "ground_exclusion", "ground_clearance", "ground_max_slope", "ground_context_px"}
_RUNTIME_NODE = {"publish_period_s", "publish_voxel_map"}
_CONFIG_FIELDS = {item.name: item for item in fields(DenseCloudConfig)}


@dataclass
class _PendingAnnotation:
    stamp: float
    frame_id: str
    source: str
    model: Any
    msg: Any = None
    payload: dict | None = None


class _Counters:
    """All diagnostic counters behind one lock (review item C34)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, float] = {}

    def add(self, name: str, amount: float = 1) -> None:
        with self._lock:
            self._values[name] = self._values.get(name, 0)+amount

    def set(self, name: str, value: float) -> None:
        with self._lock:
            self._values[name] = value

    def get(self, name: str) -> float:
        with self._lock:
            return self._values.get(name, 0)

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self._values)


class DenseCloudMappingNode(AutostartLifecycleNode):
    def __init__(self, **kwargs):
        super().__init__("dense_cloud_mapping", **kwargs)
        self._declare_parameters()
        self.pipeline: DenseCloudPipeline | None = None
        self._stats = _Counters()
        self._pipeline_lock = threading.Lock()
        self._annotation_lock = threading.Lock()
        self._result_lock = threading.Lock()
        self._active = False
        self._entities_ready = False
        self._pending_annotations: deque = deque(maxlen=1)
        self._tf_waiting: deque = deque(maxlen=1)
        self._camera_info: dict[str, CameraInfo] = {}
        self._applied_yoloe_frames: set[str] = set()
        self._last_cloud = None
        self._latest_result = None
        self._latest_cloud = None
        self._dirty = False
        self._cloud_dirty = False
        self._last_publish = -math.inf
        self._last_annotation_model = None
        self._last_processing_seconds = 0.0
        self._diag_previous = ({}, time.monotonic())
        self.tf_buffer = None
        self.tf_listener = None
        self._platform = None
        self.cloud_pub = self.map_pub = self.regions_pub = self.regions_json_pub = None
        self.cloud_sub = self.annotation_sub = self.manual_annotation_sub = self.info_sub = None
        self._tf_timer = self._publish_timer = None
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self._diagnostics = diagnostic_updater.Updater(self, period=1.0)
        self._diagnostics.setHardwareID(self.get_fully_qualified_name())
        self._diagnostics.add("dense cloud mapping", self._diagnose)

    # ------------------------------------------------------------ parameters
    def _declare_parameters(self) -> None:
        defaults = DenseCloudConfig()
        for item in fields(defaults):
            description, value_range = _CONFIG_DOCS.get(item.name, ("", None))
            value = getattr(defaults, item.name)
            # A YAML `[]` arrives untyped; dynamic typing accepts it (read back as None).
            declare(self, item.name, value, description, read_only=item.name not in _RUNTIME_CONFIG,
                    range=value_range, dynamic_typing=isinstance(value, list))
        declare(self, "cloud_topic", "points", "Input PointCloud2 (any frame with TF to world_frame)")
        declare(self, "world_frame", "map", "Fixed frame of the voxel map")
        declare(self, "camera_annotations_topic", "supermap/camera_annotations",
                "Typed supermap_msgs/CameraAnnotations input; empty disables")
        declare(self, "manual_annotations_topic", "",
                "Optional JSON std_msgs/String input for manual masks; empty (default) disables")
        declare(self, "camera_info_topic", "",
                "CameraInfo for JSON annotations that omit intrinsics; typed messages embed their own")
        declare(self, "output_cloud_topic", "supermap/dense_cloud", "Input cloud plus label fields")
        declare(self, "voxel_map_topic", "supermap/voxel_map", "World-frame voxel map (latched)")
        declare(self, "regions_topic", "supermap/regions", "supermap_msgs/RegionArray (latched)")
        declare(self, "regions_json_topic", "", "Optional JSON region metadata + stats; empty disables")
        declare(self, "publish_voxel_map", True, "Publish the voxel map", read_only=False)
        declare(self, "publish_period_s", 1.0,
                "Minimum period between voxel-map/region publications (s); 0 publishes on every update",
                read_only=False, range=(0.0, 3600.0))
        declare(self, "camera_annotation_queue_size", 512,
                "Annotations retained for clouds that have not arrived yet", range=(1, 4096))
        declare(self, "tf_timeout_s", 0.05, "Wait this long for a cloud's TF before queueing it (s)",
                range=(0.0, 5.0))
        declare(self, "tf_max_wait_s", 0.5, "Drop a queued cloud whose TF is still missing after this (s)",
                range=(0.0, 60.0))
        declare(self, "tf_queue_size", 8, "Clouds held while waiting for TF", range=(1, 256))
        declare(self, "tf_cache_s", 30.0, "TF buffer length (s)", range=(1.0, 3600.0))
        declare(self, "max_annotation_bytes", 8_000_000,
                "Reject annotation messages larger than this (JSON text or mask runs)", range=(1, 1 << 31))
        declare(self, "max_detections", 64, "Reject annotations with more masks", range=(1, 10_000))
        declare(self, "max_image_pixels", 16_777_216, "Reject annotations for larger images", range=(1, 1 << 31))
        declare(self, "max_label_length", 128, "Reject longer class names", range=(1, 4096))
        declare_platform_mask_parameters(self)

    def _param(self, name):
        return self.get_parameter(name).value

    def _on_set_parameters(self, params) -> SetParametersResult:
        changes = {p.name: p.value for p in params}
        config_changes = {}
        for name in set(changes) & _RUNTIME_CONFIG:
            value = changes[name]
            if isinstance(getattr(DenseCloudConfig(), name), float) and isinstance(value, int) \
                    and not isinstance(value, bool):
                value = float(value)
            config_changes[name] = value
        if "publish_period_s" in changes:
            value = changes["publish_period_s"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < math.inf:
                return SetParametersResult(successful=False, reason="publish_period_s must be >= 0")
        if config_changes and self.pipeline is not None:
            try:
                self.pipeline.config = replace(self.pipeline.config, **config_changes)
            except (TypeError, ValueError) as exc:
                return SetParametersResult(successful=False, reason=str(exc))
        return SetParametersResult(successful=True)

    def _config_from_parameters(self) -> DenseCloudConfig:
        values = {}
        for name, item in _CONFIG_FIELDS.items():
            value = self._param(name)
            if item.type in ("float", float) and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            if isinstance(getattr(DenseCloudConfig(), name), list):
                value = list(value or [])
            values[name] = value
        return DenseCloudConfig(**values)

    # ------------------------------------------------------------- lifecycle
    def on_configure(self, state) -> TransitionCallbackReturn:
        try:
            self._configure()
        except (ValueError, TypeError) as exc:
            self.get_logger().error(f"configure failed: {exc}")
            self._teardown()
            return TransitionCallbackReturn.FAILURE
        return TransitionCallbackReturn.SUCCESS

    def _configure(self) -> None:
        config = self._config_from_parameters()
        world_frame = str(self._param("world_frame"))
        if not world_frame:
            raise ValueError("world_frame must not be empty")
        cloud_topic = str(self._param("cloud_topic"))
        outputs = {str(self._param(name)) for name in ("output_cloud_topic", "voxel_map_topic")}
        if not cloud_topic or cloud_topic in outputs:
            raise ValueError("input cloud topic must be set and differ from output cloud topics")
        self.world_frame = world_frame
        self._limits = AnnotationLimits(
            max_detections=int(self._param("max_detections")),
            max_image_pixels=int(self._param("max_image_pixels")),
            max_label_length=int(self._param("max_label_length")))
        self._max_annotation_bytes = int(self._param("max_annotation_bytes"))
        self._tf_timeout = Duration(seconds=float(self._param("tf_timeout_s")))
        self._tf_max_wait = float(self._param("tf_max_wait_s"))
        self.pipeline = DenseCloudPipeline(config)
        self._pending_annotations = deque(maxlen=int(self._param("camera_annotation_queue_size")))
        self._tf_waiting = deque(maxlen=int(self._param("tf_queue_size")))
        # A dedicated listener thread keeps TF current while clouds are segmented.
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=float(self._param("tf_cache_s"))))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, None, spin_thread=True)
        # Masks on the robot's own body (from any annotation source) never label the map.
        self._platform = PlatformMaskSource(self)
        self._cloud_group = MutuallyExclusiveCallbackGroup()
        self._annotation_group = MutuallyExclusiveCallbackGroup()
        self._info_group = MutuallyExclusiveCallbackGroup()
        self._publish_group = MutuallyExclusiveCallbackGroup()
        stream_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.VOLATILE)
        state_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cloud_pub = self.create_lifecycle_publisher(PointCloud2, str(self._param("output_cloud_topic")),
                                                         stream_qos)
        self.map_pub = self.create_lifecycle_publisher(PointCloud2, str(self._param("voxel_map_topic")), state_qos)
        self.regions_pub = self.create_lifecycle_publisher(RegionArray, str(self._param("regions_topic")), state_qos)
        json_topic = str(self._param("regions_json_topic"))
        self.regions_json_pub = (self.create_lifecycle_publisher(String, json_topic, state_qos)
                                 if json_topic else None)
        self.cloud_sub = self.create_subscription(PointCloud2, cloud_topic, self._on_cloud, qos_profile_sensor_data,
                                                  callback_group=self._cloud_group)
        queue = self._pending_annotations.maxlen
        annotations = str(self._param("camera_annotations_topic"))
        self.annotation_sub = (self.create_subscription(CameraAnnotations, annotations, self._on_camera_annotations,
                                                        min(queue, 64), callback_group=self._annotation_group)
                               if annotations else None)
        manual = str(self._param("manual_annotations_topic"))
        self.manual_annotation_sub = (self.create_subscription(String, manual, self._on_annotations, min(queue, 64),
                                                               callback_group=self._annotation_group)
                                      if manual else None)
        info = str(self._param("camera_info_topic"))
        self.info_sub = (self.create_subscription(CameraInfo, info, self._on_camera_info, qos_profile_sensor_data,
                                                  callback_group=self._info_group)
                         if info else None)
        self._tf_timer = self.create_timer(0.05, self._retry_tf_queue, callback_group=self._cloud_group)
        self._publish_timer = self.create_timer(0.1, self._on_publish_timer, callback_group=self._publish_group)
        self._entities_ready = True
        self.get_logger().info(
            f"Full-cloud {config.input_mode} mapping configured; typed annotations on "
            f"{annotations or '(disabled)'}, manual JSON on {manual or '(disabled)'}; "
            f"allow_yoloe_labels={config.allow_yoloe_labels}; no models are loaded in the cloud node")

    def on_activate(self, state) -> TransitionCallbackReturn:
        self._active = True
        return super().on_activate(state)

    def on_deactivate(self, state) -> TransitionCallbackReturn:
        self._active = False
        return super().on_deactivate(state)

    def on_cleanup(self, state) -> TransitionCallbackReturn:
        self._teardown()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state) -> TransitionCallbackReturn:
        self._teardown()
        return TransitionCallbackReturn.SUCCESS

    def _teardown(self) -> None:
        self._active = False
        self._entities_ready = False
        for name in ("cloud_sub", "annotation_sub", "manual_annotation_sub", "info_sub"):
            sub = getattr(self, name)
            if sub is not None:
                self.destroy_subscription(sub)
                setattr(self, name, None)
        if self._platform is not None:
            self._platform.destroy()
            self._platform = None
        for name in ("_tf_timer", "_publish_timer"):
            timer = getattr(self, name)
            if timer is not None:
                self.destroy_timer(timer)
                setattr(self, name, None)
        for name in ("cloud_pub", "map_pub", "regions_pub", "regions_json_pub"):
            pub = getattr(self, name)
            if pub is not None:
                self.destroy_lifecycle_publisher(pub)
                setattr(self, name, None)
        listener, self.tf_listener = self.tf_listener, None
        if listener is not None:
            executor = getattr(listener, "executor", None)
            if executor is not None:
                executor.shutdown()
                listener.dedicated_listener_thread.join(timeout=5.0)
            try:
                listener.unregister()
                listener.node.destroy_node()
            except Exception:  # noqa: BLE001 - best effort during shutdown
                pass
        with self._pipeline_lock:
            self.pipeline = None
            self._tf_waiting.clear()
            self._last_cloud = None
        with self._annotation_lock:
            self._pending_annotations.clear()
        with self._result_lock:
            self._latest_result, self._latest_cloud, self._dirty = None, None, False

    def destroy_node(self):
        self._teardown()
        return super().destroy_node()

    # ------------------------------------------------------------------- TF
    def _world_pose(self, frame, stamp):
        if not frame:
            raise ValueError("sensor frame_id must not be empty")
        if frame == self.world_frame:
            return np.eye(4)
        if stamp.nanoseconds == 0:
            raise ValueError("a non-world-frame observation needs a nonzero capture timestamp for TF")
        return transform_to_se3(self.tf_buffer.lookup_transform(self.world_frame, frame, stamp,
                                                                timeout=self._tf_timeout))

    # ---------------------------------------------------------------- clouds
    def _on_cloud(self, message):
        if not self._active or self.pipeline is None:
            self._stats.add("clouds_dropped_inactive")
            return
        self._stats.add("clouds_received")
        with self._pipeline_lock:
            if len(self._tf_waiting) == self._tf_waiting.maxlen:
                self._stats.add("clouds_dropped_tf_queue")
                self.get_logger().warning("TF queue full; dropping the oldest cloud still waiting for TF",
                                          throttle_duration_sec=_THROTTLE)
            self._tf_waiting.append((message, time.monotonic()))
            self._process_waiting_clouds()

    def _retry_tf_queue(self):
        if not self._active or self.pipeline is None:
            return
        with self._pipeline_lock:
            if self._tf_waiting:
                self._process_waiting_clouds()

    def _process_waiting_clouds(self):
        # Caller owns the pipeline lock. Clouds are applied strictly in arrival order.
        while self._tf_waiting:
            message, arrived = self._tf_waiting[0]
            expired = time.monotonic()-arrived >= self._tf_max_wait
            if self._process_cloud(message, retry_tf=not expired) == "wait_tf":
                return
            self._tf_waiting.popleft()
        self._stats.set("clouds_waiting_for_tf", 0)

    def _process_cloud(self, message, retry_tf=False):
        started = time.perf_counter()
        try:
            if set(ANNOTATION_FIELDS) & {item.name for item in message.fields}:
                raise ValueError("incoming cloud already has segmentation fields; use the original source topic")
            try:
                transform = self._world_pose(message.header.frame_id, Time.from_msg(message.header.stamp))
            except tf2_ros.TransformException as exc:
                if retry_tf:
                    self._stats.set("clouds_waiting_for_tf", len(self._tf_waiting))
                    return "wait_tf"
                self._stats.add("tf_failures")
                raise ValueError(f"no TF {self.world_frame} <- {message.header.frame_id} at the cloud stamp: {exc}")
            xyz = full_pointcloud_xyz(message)
            result = self.pipeline.update(xyz, stamp_to_seconds(message.header.stamp), transform)
        except (ValueError, OverflowError, TypeError) as exc:
            self._stats.add("clouds_rejected")
            self.get_logger().warning(f"Cloud rejected without changing map: {exc}", throttle_duration_sec=_THROTTLE)
            return "rejected"
        except Exception as exc:  # noqa: BLE001 - one bad cloud must not stop the executor
            self._stats.add("clouds_rejected")
            self.get_logger().error(f"Cloud processing failed ({type(exc).__name__}): {exc}",
                                    throttle_duration_sec=_THROTTLE)
            return "rejected"
        self._stats.add("clouds_processed")
        segmentation = self.pipeline.last_segmentation
        self._stats.set("last_dirty_voxels", segmentation["dirty_voxels"])
        self._last_cloud = message
        self._applied_yoloe_frames.clear()
        updated = self._drain_annotations()
        if updated is not None:
            result = updated
        self._last_processing_seconds = time.perf_counter()-started
        self._stats.set("processing_seconds", self._last_processing_seconds)
        self._publish_cloud(result, message)
        self._mark_dirty(result, message)
        return "done"

    # ----------------------------------------------------------- annotations
    def _drain_annotations(self):
        # Caller owns the pipeline lock; mask reception can continue during geometry work.
        with self._annotation_lock:
            pending = list(self._pending_annotations)
            self._pending_annotations.clear()
        future = []
        matched_models = {}
        manual = []
        expired = 0
        tolerance = self.pipeline.config.max_camera_time_delta
        for entry in pending:
            delta = entry.stamp-self.pipeline.stamp
            if delta > tolerance:
                future.append(entry)
            elif abs(delta) <= tolerance:
                if entry.source == "manual":
                    manual.append(entry)
                elif entry.frame_id not in self._applied_yoloe_frames:
                    previous = matched_models.get(entry.frame_id)
                    if previous is None or abs(delta) < abs(previous.stamp-self.pipeline.stamp):
                        matched_models[entry.frame_id] = entry
            else:
                expired += 1
        with self._annotation_lock:
            arrived = list(self._pending_annotations)
            self._pending_annotations.clear()
            combined = future+arrived
            expired += max(0, len(combined)-self._pending_annotations.maxlen)
            self._pending_annotations.extend(combined)
        if expired:
            self._stats.add("annotations_expired", expired)
        result = None
        for entry in [*manual, *matched_models.values()]:
            updated = self._try_apply_annotation(entry)
            if updated is not None:
                result = updated
        return result

    def _on_camera_info(self, message):
        if not message.header.frame_id:
            self.get_logger().warning("CameraInfo without optical frame ignored", throttle_duration_sec=_THROTTLE)
            return
        self._camera_info[message.header.frame_id] = message

    def _source_allowed(self, source) -> bool:
        return source == "manual" or (source == "yoloe" and self.pipeline.config.allow_yoloe_labels)

    def _check_capture(self, frame, stamp, source):
        if not frame or not math.isfinite(stamp) or stamp <= 0:
            raise ValueError("camera needs an optical frame and a finite, positive capture time")
        if not self._source_allowed(source):
            raise ValueError("camera annotation source is not enabled")
        if source == "manual" and stamp < self.pipeline.stamp-self.pipeline.config.max_camera_time_delta:
            raise ValueError("camera labels are stale or unsynchronized with the latest cloud")

    def _on_camera_annotations(self, message):
        """Typed input: validate cheaply now, decode masks only when a matching cloud exists."""
        if not self._active or self.pipeline is None:
            return
        try:
            stamp = stamp_to_seconds(message.header.stamp)
            frame = message.header.frame_id
            self._check_capture(frame, stamp, message.source)
            if not message.rectified:
                raise ValueError("camera annotations must be rectified")
            check_rectified_camera_info(message.camera_info)
            size = self._limits.check_image(camera_info_to_intrinsics(message.camera_info))
            self._limits.check_detection_count(len(message.detections), size)
            run_bytes = 4*sum(len(item.mask_runs) for item in message.detections)
            if run_bytes > self._max_annotation_bytes:
                raise ValueError(f"mask runs total {run_bytes} bytes; max_annotation_bytes={self._max_annotation_bytes}")
            for item in message.detections:
                self._limits.check_label(item.label)
            if message.has_world_from_camera and message.world_frame not in ("", self.world_frame):
                raise ValueError(f"explicit pose is in {message.world_frame!r}, not {self.world_frame!r}")
            entry = _PendingAnnotation(stamp, frame, message.source, message.model, msg=message)
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
            self._reject_annotation(exc)
            return
        except Exception as exc:  # noqa: BLE001 - malformed input must never kill the node
            self._reject_annotation(exc)
            return
        self._enqueue(entry)

    def _on_annotations(self, message):
        """Manual JSON input (optional topic). Hardened against arbitrary payloads."""
        if not self._active or self.pipeline is None:
            return
        try:
            if len(message.data) > self._max_annotation_bytes:
                raise ValueError(f"annotation exceeds max_annotation_bytes={self._max_annotation_bytes}")
            payload = json.loads(message.data)
            if not isinstance(payload, dict):
                raise ValueError("annotation payload must be an object")
            frame, stamp, source = payload.get("frame_id"), payload.get("stamp"), payload.get("source")
            if not isinstance(frame, str) or isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
                raise ValueError("annotation needs a string frame_id and a numeric stamp")
            stamp = float(stamp)
            self._check_capture(frame, stamp, source)
            if payload.get("rectified") is not True:
                raise ValueError("camera annotations must be rectified")
            detections = payload.get("detections", [])
            if not isinstance(detections, list):
                raise ValueError("detections must be a list")
            if len(detections) > self._limits.max_detections:
                raise ValueError(f"{len(detections)} detections exceed max_detections={self._limits.max_detections}")
            payload["stamp"] = stamp
            if "intrinsics" not in payload:
                info = self._camera_info.get(frame)
                if info is None:
                    raise ValueError("provide intrinsics or matching CameraInfo in the annotation optical frame")
                info_stamp = stamp_to_seconds(info.header.stamp)
                if info_stamp and abs(info_stamp-stamp) > self.pipeline.config.max_camera_time_delta:
                    raise ValueError("CameraInfo is not synchronized with the camera observation")
                check_rectified_camera_info(info)
                intrinsics = camera_info_to_intrinsics(info)
                payload["intrinsics"] = {"fx": intrinsics.fx, "fy": intrinsics.fy, "cx": intrinsics.cx,
                                         "cy": intrinsics.cy, "width": intrinsics.width,
                                         "height": intrinsics.height}
            entry = _PendingAnnotation(stamp, frame, source, payload.get("model"), payload=payload)
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError, RecursionError) as exc:
            self._reject_annotation(exc)
            return
        self._enqueue(entry)

    def _enqueue(self, entry):
        # Reception never waits for the segmentation callback or for future TF.
        with self._annotation_lock:
            if len(self._pending_annotations) == self._pending_annotations.maxlen:
                self._stats.add("annotations_expired")
            self._pending_annotations.append(entry)
            self._stats.set("annotations_pending", len(self._pending_annotations))
        self._stats.add("annotations_received")
        if self._pipeline_lock.acquire(blocking=False):
            try:
                if self.pipeline is None:
                    return
                result = self._drain_annotations()
                if result is not None:
                    self._mark_dirty(result, self._last_cloud, cloud=True)
            finally:
                self._pipeline_lock.release()

    def _reject_annotation(self, exc):
        self._stats.add("annotations_rejected")
        self.get_logger().warning(f"Camera labels rejected; cloud geometry retained: {type(exc).__name__}: {exc}",
                                  throttle_duration_sec=_THROTTLE)

    def _try_apply_annotation(self, entry):
        try:
            allow = self.pipeline.config.allow_yoloe_labels
            if entry.msg is not None:
                stamp = Time.from_msg(entry.msg.header.stamp)
                if entry.msg.has_world_from_camera:
                    transform = _transform_msg_to_se3(entry.msg.world_from_camera)
                else:
                    transform = self._world_pose(entry.frame_id, stamp)
                camera = camera_labels_from_msg(entry.msg, T_world_from_camera=transform, allow_yoloe=allow,
                                                limits=self._limits)
            else:
                stamp = Time(nanoseconds=int(round(entry.stamp*1e9)))
                transform = self._world_pose(entry.frame_id, stamp)
                camera = camera_labels_from_dict(entry.payload, T_world_from_camera=transform, allow_yoloe=allow,
                                                 limits=self._limits)
            if self._platform.enabled:
                platform = self._platform.mask(self.tf_buffer, entry.frame_id, stamp, camera.intrinsics,
                                               timeout=self._tf_timeout)
                camera.detections, dropped = exclude_platform(camera.detections, platform, self._platform.max_overlap)
                self._stats.add("platform_detections", dropped)
            result = self.pipeline.annotate(camera)
        except PlatformUnavailable as exc:
            self._stats.add("platform_unavailable")
            self._reject_annotation(exc)
            return None
        except tf2_ros.TransformException as exc:
            self._stats.add("tf_failures")
            self._reject_annotation(exc)
            return None
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
            self._reject_annotation(exc)
            return None
        except Exception as exc:  # noqa: BLE001 - one bad mask set must not stop the executor
            self._reject_annotation(exc)
            return None
        self._stats.add("annotations_applied")
        self._last_annotation_model = entry.model
        if camera.source == "yoloe":
            self._applied_yoloe_frames.add(entry.frame_id)
        return result

    # ------------------------------------------------------------ publishing
    def _publish_cloud(self, result, message):
        if message is None or self.cloud_pub is None:
            return
        self.cloud_pub.publish(annotate_pointcloud_message(message, result))

    def _mark_dirty(self, result, message, cloud=False):
        # The labels and the cloud they index are stored together: the publish
        # timer runs concurrently with cloud processing, and an organized cloud
        # (e.g. Ouster) accepts another scan's labels without a size error.
        with self._result_lock:
            self._latest_result, self._latest_cloud, self._dirty = result, message, True
            self._cloud_dirty = cloud or self._cloud_dirty
        if float(self._param("publish_period_s")) <= 0:
            self._publish_state()

    def _on_publish_timer(self):
        if not self._active:
            return
        if time.monotonic()-self._last_publish >= float(self._param("publish_period_s")):
            self._publish_state()

    def _publish_state(self):
        """Voxel map + regions (+ a label-updated cloud), at most once per publish_period_s."""
        with self._result_lock:
            if not self._dirty or self._latest_result is None or self._latest_cloud is None:
                return
            result, cloud_dirty = self._latest_result, self._cloud_dirty
            self._dirty = self._cloud_dirty = False
            last_cloud = self._latest_cloud
        self._last_publish = time.monotonic()
        try:
            if cloud_dirty:
                self._publish_cloud(result, last_cloud)
            header = Header(stamp=last_cloud.header.stamp, frame_id=self.world_frame)
            if bool(self._param("publish_voxel_map")) and self.map_pub is not None:
                self.map_pub.publish(voxel_map_message(result, header))
            if self.regions_pub is not None:
                self.regions_pub.publish(region_array_msg(result, header))
            if self.regions_json_pub is not None:
                metadata = result_metadata(result)
                metadata.update(world_frame=self.world_frame, input_frame=last_cloud.header.frame_id,
                                counters=self._stats.snapshot(), last_annotation_model=self._last_annotation_model)
                self.regions_json_pub.publish(String(data=dumps_json(metadata)))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Publishing the map failed ({type(exc).__name__}): {exc}",
                                    throttle_duration_sec=_THROTTLE)

    # ----------------------------------------------------------- diagnostics
    def _diagnose(self, stat):
        now = time.monotonic()
        counters = self._stats.snapshot()
        with self._annotation_lock:
            counters["annotations_pending"] = len(self._pending_annotations)
        counters["clouds_waiting_for_tf"] = len(self._tf_waiting)
        previous, since = self._diag_previous
        self._diag_previous = (counters, now)
        elapsed = max(now-since, 1e-9)
        rate = (counters.get("clouds_processed", 0)-previous.get("clouds_processed", 0))/elapsed
        new_tf_failures = counters.get("tf_failures", 0)-previous.get("tf_failures", 0)
        new_drops = sum(counters.get(k, 0)-previous.get(k, 0)
                        for k in ("clouds_dropped_tf_queue", "clouds_rejected", "annotations_rejected"))
        if not self._entities_ready:
            stat.summary(DiagnosticStatus.OK, "unconfigured")
        elif not self._active:
            stat.summary(DiagnosticStatus.OK, "inactive")
        elif new_tf_failures or new_drops:
            stat.summary(DiagnosticStatus.WARN,
                         f"{new_tf_failures:g} TF failures, {new_drops:g} dropped/rejected inputs since last report")
        else:
            stat.summary(DiagnosticStatus.OK, "ok")
        stat.add("cloud_rate_hz", f"{rate:.2f}")
        stat.add("map_voxels", str(len(self.pipeline.points) if self.pipeline is not None else 0))
        for key in sorted(counters):
            stat.add(key, f"{counters[key]:g}")
        return stat


def main(args=None):
    run_node(DenseCloudMappingNode, args, multithreaded=True, num_threads=4)


if __name__ == "__main__":
    main()
