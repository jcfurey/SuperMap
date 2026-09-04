"""Live ROS2 entry point.

Subscribes to synchronized RGB, CameraInfo, PointCloud2, and Odometry
topics produced by an upstream geometric SLAM backbone (see Sec. IV-A),
runs the asynchronous open-vocabulary detector at its own rate, and drives
:class:`semantic_mapping.pipeline.SemanticMappingPipeline` on every
synchronized frame. Publishes per-object voxels, labeled 3D boxes, and an
annotated debug image, as documented in the project README.

Camera/LiDAR extrinsics and the world-from-camera pose (Eq. 3) are resolved
through TF2 rather than a single hardcoded extrinsic parameter: different
SLAM backbones publish their point cloud in different frames (world-registered
vs. sensor-frame) and provide the pose in different ways, so composing it by
hand from the Odometry message would only be correct for one specific
backbone/topology. Looking up ``world_frame -> camera_frame`` and
``camera_frame -> <point cloud frame>`` through the TF tree instead works
with any upstream SLAM backbone and any (URDF, robot_state_publisher, or
static_transform_publisher) source of the camera extrinsic -- see
``launch/semantic_mapping.launch.py`` for an optional
static_transform_publisher covering the common fixed-extrinsic case.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque

import numpy as np
import rclpy
import rclpy.time
import tf2_ros
import yaml
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import ColorRGBA, Header, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from semantic_mapping.detectors import build_detector
from semantic_mapping.geometry_utils import invert_se3, rasterize_depth, transform_points
from semantic_mapping.pipeline import FrameResult, PipelineConfig, SemanticMappingPipeline
from semantic_mapping.ros_msgs import (
    camera_info_to_intrinsics, depth_image_to_meters, image_to_numpy, numpy_to_image, pointcloud_to_xyz,
    transform_to_se3,
)
from semantic_mapping.serialization import serialize_frame
from semantic_mapping.types import CameraIntrinsics, Observation, StampedPose
from semantic_mapping.vln.clients import build_vlm_client
from semantic_mapping.vln.grounding import Grounder, GroundingRequest

_LABEL_PALETTE_SEED = 1000003  # arbitrary large prime for a stable pseudo-random per-label hue

_POINT_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
"""Memory layout of one published object point; must match the PointFields in _publish_object_points."""


def _packed_rgb_float(label: str) -> np.float32:
    """The label's colour packed as PCL's float-typed rgb field (0x00RRGGBB reinterpreted as float32)."""
    r, g, b = _label_color(label)
    packed = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
    return np.frombuffer(np.array([packed], dtype=np.uint32).tobytes(), dtype=np.float32)[0]


def _stamp_to_seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def _label_color(label: str) -> tuple[float, float, float]:
    """Deterministic, visually-distinct RGB color for a given label string."""
    h = (hash(label) * _LABEL_PALETTE_SEED) % 360
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(h / 360.0, 0.65, 0.95)
    return r, g, b


class SemanticMappingNode(Node):
    def __init__(self) -> None:
        super().__init__("semantic_mapping_node")
        self._declare_parameters()

        self.world_frame = self._param_str("world_frame", "map")
        self.camera_frame = self._param_str("camera_frame", "camera_color_optical_frame")
        self.prompts = self._load_prompts(self._param_str("prompts_file", "config/prompts.yaml"))

        self.pipeline = SemanticMappingPipeline(self._build_pipeline_config())
        self.detector = build_detector(self._param_str("detector", "offline"), **self._detector_kwargs())

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.depth_from_image = self._param_str("depth_source", "pointcloud") == "depth_image"
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self._scan_history: deque = deque(maxlen=max(int(self.get_parameter("pointcloud_accumulate_scans").value), 1))
        self._next_frame_id = 0
        self._last_detector_stamp = -float("inf")
        detector_rate_hz = float(self.get_parameter("detector_rate_hz").value)
        self._detector_period_sec = 1.0 / max(detector_rate_hz, 1e-6)
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self._publish_period_sec = 1.0 / max(publish_rate_hz, 1e-6)
        self._last_publish_stamp = -float("inf")

        # Asynchronous perception (Sec. IV, V-H): detection runs in its own
        # thread at detector_rate_hz while geometric updates continue at the
        # sensor rate. A frame handed to the detector is *deferred* -- its
        # geometry is fused together with its detections once they return,
        # under that frame's own pose and depth -- rather than fused now and
        # again later, which would double-count its geometric evidence.
        # Frames in between are fused immediately with no detections.
        self._detection_jobs: queue.Queue[Observation] = queue.Queue(maxsize=1)
        self._detection_results: queue.Queue[Observation] = queue.Queue()
        self._detector_thread = threading.Thread(target=self._detector_loop, name="detector", daemon=True)
        self._detector_thread.start()

        # Language grounding (Sec. IV-D): the query callback snapshots and
        # serializes the current graph on the executor thread (so it never
        # races the mapping callbacks), and only the model call runs here.
        self._last_result: FrameResult | None = None
        self.grounder = Grounder(
            build_vlm_client(self._param_str("vlm.client", "keyword"), **self._vlm_kwargs()),
            coordinate_frame=self.world_frame,
            local_radius_m=float(self.get_parameter("vlm.local_radius_m").value) or None,
            max_objects=int(self.get_parameter("vlm.max_objects").value) or None,
        )
        self._grounding_jobs: queue.Queue[GroundingRequest] = queue.Queue()
        self._grounding_thread = threading.Thread(target=self._grounding_loop, name="grounding", daemon=True)
        self._grounding_thread.start()

        # Persistence: restore yesterday's memory, keep today's on disk.
        self.map_save_path = self._param_str("map_save_path", "")
        load_path = self._param_str("map_load_path", "")
        if load_path:
            self.get_logger().info(self._load_map(load_path))
        autosave_sec = float(self.get_parameter("map_autosave_sec").value)
        if autosave_sec > 0 and self.map_save_path:
            self.create_timer(autosave_sec, self._autosave)
        elif autosave_sec > 0:
            self.get_logger().warning("map_autosave_sec is set but map_save_path is empty; autosave disabled")

        # Runtime accounting (Sec. V-H): module rates over each log period.
        self._stats_lock = threading.Lock()
        self._stats = {"frames": 0, "detections": 0, "publishes": 0, "stage_seconds": {}}
        self._stats_since = time.monotonic()
        stats_period = float(self.get_parameter("stats_log_period_sec").value)
        if stats_period > 0:
            self.create_timer(stats_period, self._log_runtime_stats)

        self._setup_io()
        self.get_logger().info("semantic_mapping_node initialized")

    # ------------------------------------------------------------------ setup
    def _declare_parameters(self) -> None:
        defaults: dict[str, object] = {
            "rgb_topic": "/camera/color/image_raw",
            "camera_info_topic": "/camera/color/camera_info",
            "pointcloud_topic": "/lidar/points",
            "odometry_topic": "/odometry",
            "obj_points_topic": "/obj_points",
            "obj_boxes_topic": "/obj_boxes",
            "annotated_image_topic": "/semantic_mapping/annotated_image",
            "world_frame": "map",
            "camera_frame": "camera_color_optical_frame",
            "sync_slop_sec": 0.05,
            "sync_queue_size": 30,
            "detector": "offline",
            "detector_rate_hz": 1.0,
            "prompts_file": "config/prompts.yaml",
            "publish_rate_hz": 5.0,
            "query_topic": "/semantic_mapping/query",
            "answer_topic": "/semantic_mapping/answer",
            "goal_topic": "/semantic_mapping/goal",
            "waypoints_topic": "/semantic_mapping/waypoints",
            "vlm.client": "keyword",
            "vlm.model": "",
            "vlm.base_url": "",
            "vlm.api_key_env": "",
            "vlm.local_radius_m": 0.0,
            "vlm.max_objects": 0,
            "map_load_path": "",
            "map_save_path": "",
            "map_autosave_sec": 0.0,
            "stats_log_period_sec": 10.0,
            # Sensor input: drivers commonly publish best-effort; a best-effort
            # subscription matches both reliable and best-effort publishers.
            "sensor_qos": "best_effort",   # best_effort | reliable | sensor_data
            "sensor_qos_depth": 10,
            "rgb_compressed": False,       # subscribe sensor_msgs/CompressedImage on rgb_topic
            "depth_source": "pointcloud",  # pointcloud | depth_image (aligned to the RGB camera)
            "depth_topic": "/camera/aligned_depth_to_color/image_raw",
            "depth_scale": 1000.0,         # units per meter for 16-bit depth images
            "pointcloud_accumulate_scans": 1,  # rasterize the last N scans (via TF) for sparse LiDAR
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        for name, value in PipelineConfig().__dict__.items():
            self.declare_parameter(name, value)
        self.declare_parameter("yoloe.checkpoint", "yoloe-v8l-seg.pt")
        self.declare_parameter("yoloe.device", "cuda")
        self.declare_parameter("yoloe.confidence_threshold", 0.25)
        self.declare_parameter("yoloe.sam2_checkpoint", "")
        self.declare_parameter("yoloe.sam2_model_cfg", "")
        self.declare_parameter("groundingdino.config_path", "")
        self.declare_parameter("groundingdino.checkpoint_path", "")
        self.declare_parameter("groundingdino.sam2_checkpoint", "")
        self.declare_parameter("groundingdino.sam2_model_cfg", "")
        self.declare_parameter("groundingdino.device", "cuda")
        self.declare_parameter("groundingdino.box_threshold", 0.35)
        self.declare_parameter("groundingdino.text_threshold", 0.25)
        self.declare_parameter("offline.detections_dir", "")

    def _param_str(self, name: str, default: str) -> str:
        value = self.get_parameter(name).value
        return default if value is None else str(value)

    def _load_prompts(self, prompts_file: str) -> list[str]:
        try:
            with open(prompts_file) as f:
                data = yaml.safe_load(f) or {}
            return list(data.get("prompts", []))
        except OSError:
            self.get_logger().warning(f"Could not read prompts file '{prompts_file}', using empty vocabulary")
            return []

    def _build_pipeline_config(self) -> PipelineConfig:
        kwargs = {}
        for name in PipelineConfig().__dict__:
            kwargs[name] = self.get_parameter(name).value
        return PipelineConfig(**kwargs)

    def _detector_kwargs(self) -> dict:
        backend = self._param_str("detector", "offline")
        if backend == "yoloe":
            return {
                "checkpoint": self._param_str("yoloe.checkpoint", "yoloe-v8l-seg.pt"),
                "device": self._param_str("yoloe.device", "cuda"),
                "confidence_threshold": float(self.get_parameter("yoloe.confidence_threshold").value),
                "sam2_checkpoint": self._param_str("yoloe.sam2_checkpoint", "") or None,
                "sam2_model_cfg": self._param_str("yoloe.sam2_model_cfg", "") or None,
            }
        if backend in ("groundingdino", "grounding_dino", "gdino"):
            return {
                "config_path": self._param_str("groundingdino.config_path", ""),
                "checkpoint_path": self._param_str("groundingdino.checkpoint_path", ""),
                "sam2_checkpoint": self._param_str("groundingdino.sam2_checkpoint", "") or None,
                "sam2_model_cfg": self._param_str("groundingdino.sam2_model_cfg", "") or None,
                "device": self._param_str("groundingdino.device", "cuda"),
                "box_threshold": float(self.get_parameter("groundingdino.box_threshold").value),
                "text_threshold": float(self.get_parameter("groundingdino.text_threshold").value),
            }
        return {"detections_dir": self._param_str("offline.detections_dir", "")}

    def _vlm_kwargs(self) -> dict:
        """Only pass what was configured, so each backend keeps its own defaults."""
        kwargs = {}
        for key in ("model", "base_url", "api_key_env"):
            value = self._param_str(f"vlm.{key}", "")
            if value:
                kwargs[key] = value
        return kwargs

    def _sensor_qos(self) -> QoSProfile:
        name = self._param_str("sensor_qos", "best_effort").lower()
        if name == "sensor_data":
            return qos_profile_sensor_data
        reliability = ReliabilityPolicy.RELIABLE if name == "reliable" else ReliabilityPolicy.BEST_EFFORT
        return QoSProfile(reliability=reliability, history=HistoryPolicy.KEEP_LAST,
                          depth=int(self.get_parameter("sensor_qos_depth").value))

    def _setup_io(self) -> None:
        qos = self._sensor_qos()
        rgb_type = CompressedImage if bool(self.get_parameter("rgb_compressed").value) else Image
        rgb_sub = Subscriber(self, rgb_type, self._param_str("rgb_topic", "/camera/color/image_raw"), qos_profile=qos)
        info_sub = Subscriber(self, CameraInfo, self._param_str("camera_info_topic", "/camera/color/camera_info"),
                              qos_profile=qos)
        if self.depth_from_image:
            depth_sub = Subscriber(self, Image, self._param_str("depth_topic", "/camera/aligned_depth_to_color/image_raw"),
                                   qos_profile=qos)
        else:
            depth_sub = Subscriber(self, PointCloud2, self._param_str("pointcloud_topic", "/lidar/points"),
                                   qos_profile=qos)
        odom_sub = Subscriber(self, Odometry, self._param_str("odometry_topic", "/odometry"), qos_profile=qos)
        self._sensor_subscribers = [rgb_sub, info_sub, depth_sub, odom_sub]

        self._sync = ApproximateTimeSynchronizer(
            [rgb_sub, info_sub, depth_sub, odom_sub],
            queue_size=int(self.get_parameter("sync_queue_size").value),
            slop=float(self.get_parameter("sync_slop_sec").value),
        )
        self._sync.registerCallback(self._on_synced_frame)

        self.obj_points_pub = self.create_publisher(
            PointCloud2, self._param_str("obj_points_topic", "/obj_points"), 10)
        self.obj_boxes_pub = self.create_publisher(
            MarkerArray, self._param_str("obj_boxes_topic", "/obj_boxes"), 10)
        self.annotated_image_pub = self.create_publisher(
            Image, self._param_str("annotated_image_topic", "/semantic_mapping/annotated_image"), 10)

        self.query_sub = self.create_subscription(
            String, self._param_str("query_topic", "/semantic_mapping/query"), self._on_query, 10)
        self.answer_pub = self.create_publisher(
            String, self._param_str("answer_topic", "/semantic_mapping/answer"), 10)
        self.goal_pub = self.create_publisher(
            PoseStamped, self._param_str("goal_topic", "/semantic_mapping/goal"), 10)
        self.waypoints_pub = self.create_publisher(
            Path, self._param_str("waypoints_topic", "/semantic_mapping/waypoints"), 10)

        self.save_map_srv = self.create_service(Trigger, "~/save_map", self._on_save_map)
        self.load_map_srv = self.create_service(Trigger, "~/load_map", self._on_load_map)

    # ----------------------------------------------------------- persistence
    def _save_map(self, path: str) -> str:
        """Save the map; returns a human-readable outcome. Runs on the executor
        thread, like every map update, so it never races the pipeline."""
        saved = self.pipeline.save(path, metadata={"world_frame": self.world_frame})
        return f"saved {len(self.pipeline.object_map.objects)} instances to {saved}"

    def _load_map(self, path: str) -> str:
        header = self.pipeline.load(path)
        self._last_result = None  # the next frame re-derives the graph from the restored map
        return f"restored {header['num_instances']} instances from {path}"

    def _on_save_map(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if not self.map_save_path:
            response.success, response.message = False, "map_save_path parameter is empty"
            return response
        try:
            response.success, response.message = True, self._save_map(self.map_save_path)
            self.get_logger().info(response.message)
        except Exception as exc:  # report to the caller instead of killing the executor
            response.success, response.message = False, f"save failed: {exc}"
            self.get_logger().error(response.message)
        return response

    def _on_load_map(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        path = self._param_str("map_load_path", "") or self.map_save_path
        if not path:
            response.success, response.message = False, "neither map_load_path nor map_save_path is set"
            return response
        try:
            response.success, response.message = True, self._load_map(path)
            self.get_logger().info(response.message)
        except Exception as exc:
            response.success, response.message = False, f"load failed: {exc}"
            self.get_logger().error(response.message)
        return response

    def _autosave(self) -> None:
        try:
            self.get_logger().debug(self._save_map(self.map_save_path))
        except Exception as exc:
            self.get_logger().error(f"autosave failed: {exc}")

    def _log_runtime_stats(self) -> None:
        """Report module rates in the terms of Sec. V-H: 2D segmentation (detector),
        3D mapping (geometric update at the sensor rate), and 4D scene graph output."""
        now = time.monotonic()
        with self._stats_lock:
            elapsed = max(now - self._stats_since, 1e-6)
            frames, detections, publishes = (self._stats[k] for k in ("frames", "detections", "publishes"))
            stage_seconds = dict(self._stats["stage_seconds"])
            self._stats = {"frames": 0, "detections": 0, "publishes": 0, "stage_seconds": {}}
            self._stats_since = now
        if frames == 0 and detections == 0:
            return
        stages = " ".join(f"{k}={1e3 * v / max(frames, 1):.1f}ms" for k, v in stage_seconds.items())
        objects = len(self.pipeline.object_map.objects)
        self.get_logger().info(
            f"runtime: detector {detections / elapsed:.2f} Hz, 3D mapping {frames / elapsed:.2f} Hz, "
            f"scene graph published {publishes / elapsed:.2f} Hz, {objects} instances | per frame: {stages}"
        )

    # ------------------------------------------------------------- grounding
    def _on_query(self, msg: String) -> None:
        """Instruction in -> (asynchronously) waypoints out (Sec. IV-D)."""
        instruction = msg.data.strip()
        if not instruction:
            return
        if self._last_result is None:
            self.get_logger().warning(f"query '{instruction}' received before any map update; ignoring")
            return
        robot_position = None
        try:
            T = self._lookup_se3(self.world_frame, self.camera_frame, self.get_clock().now().to_msg())
            robot_position = T[:3, 3]
        except tf2_ros.TransformException:
            pass  # local-subgraph selection just falls back to the whole graph
        request = self.grounder.prepare(
            instruction, self._last_result.objects, self._last_result.scene_graph, robot_position,
        )
        self._grounding_jobs.put(request)

    def _grounding_loop(self) -> None:
        while rclpy.ok():
            try:
                request = self._grounding_jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            result = self.grounder.complete(request)
            if result.error:
                self.get_logger().warning(f"grounding '{request.instruction}': {result.error}")
            else:
                self.get_logger().info(
                    f"grounding '{request.instruction}' -> instances {result.target_ids}")

            header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.world_frame)
            self.answer_pub.publish(String(data=json.dumps(result.to_dict())))
            if not result.waypoints:
                continue
            path = Path(header=header)
            for waypoint in result.waypoints:
                pose = PoseStamped(header=header)
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = (float(v) for v in waypoint)
                pose.pose.orientation.w = 1.0
                path.poses.append(pose)
            self.waypoints_pub.publish(path)
            self.goal_pub.publish(path.poses[0])

    # ---------------------------------------------------------------- callback
    def _on_synced_frame(self, rgb_msg, info_msg: CameraInfo, depth_msg, odom_msg: Odometry) -> None:
        # odom_msg's own pose fields are not read directly: a well-behaved
        # SLAM backbone also broadcasts the same pose as a dynamic TF
        # transform, so resolving world_frame -> camera_frame (and
        # <point cloud frame> -> camera_frame) through TF2 handles the full
        # chain -- dynamic odometry composed with whatever static camera
        # extrinsic is in the TF tree -- without this node hardcoding either.
        # The message is still subscribed to keep the four-topic sync
        # documented in the README and to pace processing at the SLAM
        # backbone's pose-update rate.
        try:
            T_world_from_cam = self._lookup_se3(self.world_frame, self.camera_frame, rgb_msg.header.stamp)
            T_cam_from_cloud = None if self.depth_from_image else self._lookup_se3(
                self.camera_frame, depth_msg.header.frame_id, depth_msg.header.stamp)
        except tf2_ros.TransformException as exc:
            self.get_logger().warning(
                f"TF lookup failed, skipping frame: {exc}", throttle_duration_sec=5.0)
            return

        stamp = _stamp_to_seconds(rgb_msg.header.stamp)
        intrinsics = camera_info_to_intrinsics(info_msg)
        try:
            rgb = self._decode_rgb(rgb_msg, intrinsics)
        except ValueError as exc:
            self.get_logger().error(f"cannot decode RGB image: {exc}", throttle_duration_sec=5.0)
            return

        if self.depth_from_image:
            # Depth already aligned to the RGB camera (e.g. an RGB-D driver's aligned stream).
            depth = depth_image_to_meters(depth_msg, self.depth_scale)
            if depth.shape != (intrinsics.height, intrinsics.width):
                self.get_logger().error(
                    f"depth image {depth.shape[1]}x{depth.shape[0]} does not match CameraInfo "
                    f"{intrinsics.width}x{intrinsics.height}; use the driver's depth-aligned-to-color stream",
                    throttle_duration_sec=5.0)
                return
        else:
            # Rasterize the synchronized point cloud into the camera frame to
            # obtain the raw sensor depth D(u) needed by geometric consistency.
            points_cloud_frame = pointcloud_to_xyz(depth_msg)
            points_cam = transform_points(T_cam_from_cloud, points_cloud_frame) if points_cloud_frame.shape[0] \
                else points_cloud_frame
            if self._scan_history.maxlen > 1:
                # A single sparse LiDAR scan covers few pixels; the last N scans,
                # carried in the world frame and re-projected under the current
                # pose, give the consistency update a denser depth image.
                self._scan_history.append(transform_points(T_world_from_cam, points_cam))
                points_cam = transform_points(
                    invert_se3(T_world_from_cam), np.concatenate(list(self._scan_history), axis=0))
            depth = rasterize_depth(points_cam, intrinsics.K, intrinsics.width, intrinsics.height)

        observation = Observation(
            stamp=stamp,
            pose=StampedPose(stamp=stamp, T_world_from_frame=T_world_from_cam),
            intrinsics=intrinsics,
            frame_id=self._next_frame_id,
            rgb=rgb,
            depth=depth,
            detections=[],
        )
        self._next_frame_id += 1

        # Fuse any detection frames the worker finished since last time (each
        # under its own buffered pose/depth), then decide what to do with this one.
        self._drain_detection_results(rgb_msg.header)

        detection_due = stamp - self._last_detector_stamp >= self._detector_period_sec
        if detection_due and not self._detection_jobs.full():
            self._detection_jobs.put(observation)
            self._last_detector_stamp = stamp
            return  # deferred: processed in _drain_detection_results once detections arrive

        self._process_and_publish(observation, rgb_msg.header)

    def _detector_loop(self) -> None:
        while rclpy.ok():
            try:
                observation = self._detection_jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                observation.detections = self.detector.detect(
                    observation.rgb,
                    prompts=self.prompts,
                    frame_id=observation.frame_id,
                )
            except Exception as exc:  # noqa: BLE001 - a detector failure must not kill the mapping loop
                self.get_logger().error(f"detector failed, fusing frame without detections: {exc}",
                                        throttle_duration_sec=5.0)
                observation.detections = []
            with self._stats_lock:
                self._stats["detections"] += 1
            self._detection_results.put(observation)

    def _drain_detection_results(self, header: Header) -> None:
        while True:
            try:
                observation = self._detection_results.get_nowait()
            except queue.Empty:
                return
            result = self._process_and_publish(observation, header)
            self._publish_annotated_image(observation, result, header)

    def _process_and_publish(self, observation: Observation, header: Header) -> FrameResult:
        result = self.pipeline.process_frame(observation)
        self._last_result = result
        with self._stats_lock:
            self._stats["frames"] += 1
            for stage, seconds in result.timings.items():
                self._stats["stage_seconds"][stage] = self._stats["stage_seconds"].get(stage, 0.0) + seconds
        if observation.stamp - self._last_publish_stamp >= self._publish_period_sec:
            self._publish_result(result, header)
            self._last_publish_stamp = observation.stamp
            with self._stats_lock:
                self._stats["publishes"] += 1
        return result

    def _lookup_se3(self, target_frame: str, source_frame: str, stamp) -> np.ndarray:
        """4x4 SE(3) transform mapping ``source_frame``-expressed points/poses
        into ``target_frame`` coordinates, per REP 105 TF2 lookup semantics.
        """
        return transform_to_se3(self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time.from_msg(stamp)))

    def _decode_rgb(self, rgb_msg, intrinsics: CameraIntrinsics) -> np.ndarray:
        """RGB (H, W, 3) at the CameraInfo resolution from an Image or CompressedImage."""
        rgb = image_to_numpy(rgb_msg)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[:, :, None], 3, axis=2)
        if rgb.shape[:2] != (intrinsics.height, intrinsics.width):
            import cv2

            rgb = cv2.resize(rgb, (intrinsics.width, intrinsics.height), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(rgb)

    # ---------------------------------------------------------------- publish
    def _publish_annotated_image(self, observation: Observation, result: FrameResult, header: Header) -> None:
        if observation.rgb is None or self.annotated_image_pub.get_subscription_count() == 0:
            return
        try:
            import cv2
        except ImportError:
            self.get_logger().warning("opencv-python not installed; annotated image disabled", once=True)
            return

        image = np.ascontiguousarray(observation.rgb.copy())
        for detection, instance_id in zip(observation.detections, result.detection_instance_ids):
            r, g, b = (int(c * 255) for c in _label_color(detection.label))
            x1, y1, x2, y2 = (int(round(v)) for v in detection.bbox)
            if detection.mask is not None:
                image[detection.mask] = (0.6 * image[detection.mask] + 0.4 * np.array([r, g, b])).astype(np.uint8)
            cv2.rectangle(image, (x1, y1), (x2, y2), (r, g, b), 2)
            tag = f"#{instance_id} {detection.label} {detection.score:.2f}" if instance_id >= 0 \
                else f"{detection.label} {detection.score:.2f} (dropped)"
            cv2.putText(image, tag, (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (r, g, b), 1, cv2.LINE_AA)

        self.annotated_image_pub.publish(numpy_to_image(image, "rgb8", header))

    def _publish_result(self, result: FrameResult, header: Header) -> None:
        header.frame_id = self.world_frame
        self._publish_object_points(result, header)
        self._publish_object_boxes(result, header)
        # The per-frame JSON schema (bbox3d, label, id, center, spatial_relations,
        # status, latest_stamp) is available to any downstream consumer via:
        #   serialize_frame(result.objects, result.scene_graph)
        # and is intentionally not published as a ROS message here, keeping the
        # wire schema identical between offline and live modes (see README).
        _ = serialize_frame(result.objects, result.scene_graph)

    def _publish_object_points(self, result: FrameResult, header: Header) -> None:
        fields = [
            pc2.PointField(name="x", offset=0, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name="y", offset=4, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name="z", offset=8, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name="rgb", offset=12, datatype=pc2.PointField.FLOAT32, count=1),
        ]
        # One structured array for the whole map, serialized by create_cloud in
        # a single copy: a Python row per point took 140 ms for a room-sized
        # map and over a second for a building (doc/audit-2026-09.md, P1).
        counts = [obj.points_world.shape[0] for obj in result.objects]
        cloud = np.zeros(int(sum(counts)), dtype=_POINT_DTYPE)
        offset = 0
        for obj, n in zip(result.objects, counts):
            if n == 0:
                continue
            block = cloud[offset:offset + n]
            block["x"] = obj.points_world[:, 0]
            block["y"] = obj.points_world[:, 1]
            block["z"] = obj.points_world[:, 2]
            block["rgb"] = _packed_rgb_float(obj.label)
            offset += n
        self.obj_points_pub.publish(pc2.create_cloud(header, fields, cloud))

    def _publish_object_boxes(self, result: FrameResult, header: Header) -> None:
        marker_array = MarkerArray()
        node_ids = set(result.scene_graph.node_ids) if result.scene_graph else set()
        by_id = {obj.instance_id: obj for obj in result.objects}

        for marker_index, instance_id in enumerate(sorted(node_ids)):
            obj = by_id.get(instance_id)
            if obj is None:
                continue
            r, g, b = _label_color(obj.label)
            xmin, ymin, zmin, xmax, ymax, zmax = obj.bbox3d

            box = Marker()
            box.header = header
            box.ns = "obj_boxes"
            box.id = marker_index * 2
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
            label_marker.id = marker_index * 2 + 1
            label_marker.type = Marker.TEXT_VIEW_FACING
            label_marker.action = Marker.ADD
            label_marker.pose.position.x = box.pose.position.x
            label_marker.pose.position.y = box.pose.position.y
            label_marker.pose.position.z = float(zmax + 0.1)
            label_marker.scale.z = 0.15
            label_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            label_marker.text = f"{instance_id}:{obj.label} ({obj.status.value})"
            marker_array.markers.append(label_marker)

        self.obj_boxes_pub.publish(marker_array)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SemanticMappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
