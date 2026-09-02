"""Live ROS2 entry point.

Subscribes to synchronized RGB, CameraInfo, PointCloud2, and Odometry
topics produced by an upstream geometric SLAM backbone (SuperOdometry, see
Sec. IV-A), runs the asynchronous open-vocabulary detector at its own rate,
and drives :class:`semantic_mapping.pipeline.SemanticMappingPipeline` on
every synchronized frame. Publishes per-object voxels, labeled 3D boxes, and
an annotated debug image, as documented in the project README.
"""
from __future__ import annotations

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from semantic_mapping.detectors import build_detector
from semantic_mapping.geometry_utils import (
    invert_se3,
    rasterize_depth,
    se3_from_translation_quaternion,
    transform_points,
)
from semantic_mapping.pipeline import FrameResult, PipelineConfig, SemanticMappingPipeline
from semantic_mapping.serialization import serialize_frame
from semantic_mapping.types import CameraIntrinsics, Observation, StampedPose

_LABEL_PALETTE_SEED = 1000003  # arbitrary large prime for a stable pseudo-random per-label hue


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
        self.prompts = self._load_prompts(self._param_str("prompts_file", "config/prompts.yaml"))

        self.pipeline = SemanticMappingPipeline(self._build_pipeline_config())
        self.detector = build_detector(self._param_str("detector", "offline"), **self._detector_kwargs())

        translation = np.array(self.get_parameter("body_from_camera_extrinsics").value[:3], dtype=np.float64)
        quaternion = np.array(self.get_parameter("body_from_camera_extrinsics").value[3:], dtype=np.float64)
        self.T_body_from_cam = se3_from_translation_quaternion(translation, quaternion)
        self.T_cam_from_body = invert_se3(self.T_body_from_cam)

        self.bridge = CvBridge()
        self._last_detector_stamp = -float("inf")
        detector_rate_hz = float(self.get_parameter("detector_rate_hz").value)
        self._detector_period_sec = 1.0 / max(detector_rate_hz, 1e-6)

        self._setup_io()
        self.get_logger().info("semantic_mapping_node initialized")

    # ------------------------------------------------------------------ setup
    def _declare_parameters(self) -> None:
        defaults: dict[str, object] = {
            "rgb_topic": "/camera/color/image_raw",
            "camera_info_topic": "/camera/color/camera_info",
            "pointcloud_topic": "/lidar/points",
            "odometry_topic": "/super_odometry/odometry",
            "obj_points_topic": "/obj_points",
            "obj_boxes_topic": "/obj_boxes",
            "annotated_image_topic": "/semantic_mapping/annotated_image",
            "world_frame": "map",
            "body_frame": "base_link",
            "camera_frame": "camera_color_optical_frame",
            "body_from_camera_extrinsics": [0.0, 0.0, 0.0, -0.5, 0.5, -0.5, 0.5],
            "sync_slop_sec": 0.05,
            "sync_queue_size": 30,
            "detector": "offline",
            "detector_rate_hz": 1.0,
            "prompts_file": "config/prompts.yaml",
            "publish_rate_hz": 5.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        for name, value in PipelineConfig().__dict__.items():
            self.declare_parameter(name, value)
        self.declare_parameter("yoloe.checkpoint", "yoloe-v8l-seg.pt")
        self.declare_parameter("yoloe.device", "cuda")
        self.declare_parameter("yoloe.confidence_threshold", 0.25)
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
            self.get_logger().warn(f"Could not read prompts file '{prompts_file}', using empty vocabulary")
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

    def _setup_io(self) -> None:
        rgb_sub = Subscriber(self, Image, self._param_str("rgb_topic", "/camera/color/image_raw"))
        info_sub = Subscriber(self, CameraInfo, self._param_str("camera_info_topic", "/camera/color/camera_info"))
        pc_sub = Subscriber(self, PointCloud2, self._param_str("pointcloud_topic", "/lidar/points"))
        odom_sub = Subscriber(self, Odometry, self._param_str("odometry_topic", "/super_odometry/odometry"))

        self._sync = ApproximateTimeSynchronizer(
            [rgb_sub, info_sub, pc_sub, odom_sub],
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

    # ---------------------------------------------------------------- callback
    def _on_synced_frame(self, rgb_msg: Image, info_msg: CameraInfo, pc_msg: PointCloud2,
                          odom_msg: Odometry) -> None:
        stamp = _stamp_to_seconds(rgb_msg.header.stamp)

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        intrinsics = CameraIntrinsics(
            fx=info_msg.k[0], fy=info_msg.k[4], cx=info_msg.k[2], cy=info_msg.k[5],
            width=info_msg.width, height=info_msg.height,
        )

        body_translation = np.array([
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            odom_msg.pose.pose.position.z,
        ])
        body_quaternion = np.array([
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
            odom_msg.pose.pose.orientation.w,
        ])
        T_world_from_body = se3_from_translation_quaternion(body_translation, body_quaternion)
        # Eq. (3): P_t = T_world_from_body * T_BC
        T_world_from_cam = T_world_from_body @ self.T_body_from_cam

        # The synchronized point cloud is expected pre-registered to body_frame
        # by the upstream SLAM backbone; rasterize it into the camera frame to
        # obtain the raw sensor depth D(u) used by geometric consistency.
        points_body = self._read_pointcloud_xyz(pc_msg)
        points_cam = transform_points(self.T_cam_from_body, points_body) if points_body.shape[0] else points_body
        depth = rasterize_depth(points_cam, intrinsics.K, intrinsics.width, intrinsics.height)

        detections = []
        if stamp - self._last_detector_stamp >= self._detector_period_sec:
            detections = self.detector.detect(rgb, prompts=self.prompts)
            self._last_detector_stamp = stamp

        observation = Observation(
            stamp=stamp,
            pose=StampedPose(stamp=stamp, T_world_from_frame=T_world_from_cam),
            intrinsics=intrinsics,
            rgb=rgb,
            depth=depth,
            detections=detections,
        )

        result = self.pipeline.process_frame(observation)
        self._publish_result(result, rgb_msg.header)

    @staticmethod
    def _read_pointcloud_xyz(pc_msg: PointCloud2) -> np.ndarray:
        # read_points returns a structured numpy array; stay vectorized (no
        # per-point Python loop) since a LiDAR scan can carry tens of
        # thousands of points and this runs every synchronized frame.
        cloud = pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True)
        if cloud.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=-1).astype(np.float64)

    # ---------------------------------------------------------------- publish
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
        rows = []
        for obj in result.objects:
            if obj.points_world.shape[0] == 0:
                continue
            r, g, b = _label_color(obj.label)
            packed_rgb = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
            packed_rgb_float = np.frombuffer(np.array([packed_rgb], dtype=np.uint32).tobytes(), dtype=np.float32)[0]
            for point in obj.points_world:
                rows.append([point[0], point[1], point[2], packed_rgb_float])

        cloud_msg = pc2.create_cloud(header, fields, rows)
        self.obj_points_pub.publish(cloud_msg)

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
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
