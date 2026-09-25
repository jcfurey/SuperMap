"""ROS 2 full-cloud mapping, independent of camera availability and model inference."""
from __future__ import annotations

from dataclasses import asdict, fields
from collections import deque
import json
import time
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, PointCloud2
from std_msgs.msg import Header, String
import tf2_ros

from semantic_mapping.dense_cloud import DenseCloudConfig, DenseCloudPipeline
from semantic_mapping.dense_cloud_io import (
    ANNOTATION_FIELDS, annotate_pointcloud_message, camera_labels_from_dict, full_pointcloud_xyz,
    result_metadata, voxel_map_message,
)
from semantic_mapping.ros_msgs import camera_info_to_intrinsics, stamp_to_seconds, transform_to_se3


class DenseCloudMappingNode(Node):
    def __init__(self):
        super().__init__("dense_cloud_mapping")
        defaults = DenseCloudConfig()
        for item in fields(defaults):
            self.declare_parameter(item.name, getattr(defaults, item.name))
        for name, value in {
            "cloud_topic": "/points", "world_frame": "map",
            "camera_annotations_topic": "/supermap/camera_annotations",
            "camera_info_topic": "", "output_cloud_topic": "/supermap/dense_cloud",
            "voxel_map_topic": "/supermap/voxel_map", "regions_topic": "/supermap/regions",
            "publish_voxel_map": True,
            "camera_annotation_queue_size": 512,
        }.items():
            self.declare_parameter(name, value)
        config = DenseCloudConfig(**{item.name: self.get_parameter(item.name).value for item in fields(defaults)})
        self.pipeline = DenseCloudPipeline(config)
        self.world_frame = str(self.get_parameter("world_frame").value)
        if not self.world_frame:
            raise ValueError("world_frame must not be empty")
        # Accumulated maps may be published seconds after their last observation.
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._pipeline_lock = threading.Lock()
        self._annotation_lock = threading.Lock()
        self._annotation_group = MutuallyExclusiveCallbackGroup()
        self._last_cloud = None
        self._camera_info = {}
        self._last_processing_seconds = 0.0
        self._rejected_clouds = 0
        self._rejected_annotations = 0
        self._applied_annotations = 0
        self._expired_annotations = 0
        self._applied_yoloe_frames = set()
        self._last_annotation_model = None
        queue_size = self.get_parameter("camera_annotation_queue_size").value
        if type(queue_size) is not int or not 1 <= queue_size <= 4096:
            raise ValueError("camera_annotation_queue_size must be an integer in [1, 4096]")
        self._pending_annotations = deque(maxlen=queue_size)
        state_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cloud_pub = self.create_publisher(PointCloud2, self._topic("output_cloud_topic"), 2)
        self.map_pub = self.create_publisher(PointCloud2, self._topic("voxel_map_topic"), state_qos)
        self.regions_pub = self.create_publisher(String, self._topic("regions_topic"), state_qos)
        cloud_topic = self._topic("cloud_topic")
        if cloud_topic in {self._topic("output_cloud_topic"), self._topic("voxel_map_topic")}:
            raise ValueError("input cloud topic must differ from output cloud topics")
        self.cloud_sub = self.create_subscription(PointCloud2, cloud_topic, self._on_cloud, qos_profile_sensor_data)
        annotations = self._topic("camera_annotations_topic")
        self.annotation_sub = (self.create_subscription(String, annotations, self._on_annotations, queue_size,
                                                          callback_group=self._annotation_group)
                               if annotations else None)
        info = self._topic("camera_info_topic")
        self.info_sub = (self.create_subscription(CameraInfo, info, self._on_camera_info, qos_profile_sensor_data)
                        if info else None)
        self.get_logger().info(
            f"Full-cloud {config.input_mode} mapping initialized; optional external camera labels; "
            f"allow_yoloe_labels={config.allow_yoloe_labels}; no models are loaded in the cloud node")

    def _topic(self, name):
        return str(self.get_parameter(name).value)

    def _world_pose(self, frame, stamp):
        if not frame:
            raise ValueError("sensor frame_id must not be empty")
        if frame == self.world_frame:
            return np.eye(4)
        if stamp.nanoseconds == 0:
            raise ValueError("a non-world-frame observation needs a nonzero capture timestamp for TF")
        return transform_to_se3(self.tf_buffer.lookup_transform(self.world_frame, frame, stamp))

    def _on_cloud(self, message):
        with self._pipeline_lock:
            self._process_cloud(message)

    def _process_cloud(self, message):
        started = time.perf_counter()
        try:
            if set(ANNOTATION_FIELDS) & {item.name for item in message.fields}:
                raise ValueError("incoming cloud already has segmentation fields; use the original source topic")
            transform = self._world_pose(message.header.frame_id, Time.from_msg(message.header.stamp))
            xyz = full_pointcloud_xyz(message)
            result = self.pipeline.update(xyz, stamp_to_seconds(message.header.stamp), transform)
        except (ValueError, OverflowError, tf2_ros.TransformException) as exc:
            self._rejected_clouds += 1
            self.get_logger().warning(f"Cloud rejected without changing map: {exc}")
            return
        self._last_cloud = message
        self._applied_yoloe_frames.clear()
        updated = self._drain_annotations()
        if updated is not None:
            result = updated
        self._last_processing_seconds = time.perf_counter()-started
        self._publish(result)

    def _drain_annotations(self):
        # Caller owns the pipeline lock; mask reception can continue during geometry work.
        with self._annotation_lock:
            pending = list(self._pending_annotations)
            self._pending_annotations.clear()
        future = []
        matched_models = {}
        manual = []
        tolerance = self.pipeline.config.max_camera_time_delta
        for payload in pending:
            delta = payload["stamp"]-self.pipeline.stamp
            if delta > tolerance:
                future.append(payload)
            elif abs(delta) <= tolerance:
                if payload["source"] == "manual":
                    manual.append(payload)
                elif payload["frame_id"] not in self._applied_yoloe_frames:
                    frame = payload["frame_id"]
                    previous = matched_models.get(frame)
                    if previous is None or abs(delta) < abs(previous["stamp"]-self.pipeline.stamp):
                        matched_models[frame] = payload
            else:
                self._expired_annotations += 1
        with self._annotation_lock:
            arrived = list(self._pending_annotations)
            self._pending_annotations.clear()
            combined = future+arrived
            self._expired_annotations += max(0, len(combined)-self._pending_annotations.maxlen)
            self._pending_annotations.extend(combined)
        result = None
        for payload in [*manual, *matched_models.values()]:
            updated = self._try_apply_annotation(payload)
            if updated is not None:
                result = updated
        return result

    def _on_camera_info(self, message):
        if not message.header.frame_id:
            self.get_logger().warning("CameraInfo without optical frame ignored")
            return
        self._camera_info[message.header.frame_id] = message

    def _on_annotations(self, message):
        try:
            payload = json.loads(message.data)
            if not isinstance(payload, dict):
                raise ValueError("annotation payload must be an object")
            frame = str(payload["frame_id"])
            stamp = float(payload["stamp"])
            if not frame or not np.isfinite(stamp) or stamp < 0:
                raise ValueError("camera needs an optical frame and finite nonnegative capture time")
            source = payload.get("source")
            if source != "manual" and not (source == "yoloe" and self.pipeline.config.allow_yoloe_labels):
                raise ValueError("camera annotation source is not enabled")
            if payload.get("rectified") is not True:
                raise ValueError("camera annotations must be rectified")
            payload["stamp"] = stamp
            if "intrinsics" not in payload:
                info = self._camera_info.get(frame)
                if info is None:
                    raise ValueError("provide intrinsics or matching CameraInfo in the annotation optical frame")
                info_stamp = stamp_to_seconds(info.header.stamp)
                if info_stamp and abs(info_stamp-stamp) > self.pipeline.config.max_camera_time_delta:
                    raise ValueError("CameraInfo is not synchronized with the camera observation")
                if not np.isfinite(info.d).all() or any(abs(value) > 1e-12 for value in info.d):
                    raise ValueError("distorted CameraInfo cannot describe rectified masks")
                rotation = np.asarray(info.r).reshape(3, 3)
                if rotation.any() and not np.allclose(rotation, np.eye(3)):
                    raise ValueError("nonidentity camera rectification requires explicit rectified optical calibration")
                payload["intrinsics"] = asdict(camera_info_to_intrinsics(info))
            if (source == "manual" and stamp < self.pipeline.stamp-self.pipeline.config.max_camera_time_delta):
                raise ValueError("camera labels are stale or unsynchronized with the latest cloud")
            # Reception never waits for the segmentation callback or for future TF.
            with self._annotation_lock:
                if len(self._pending_annotations) == self._pending_annotations.maxlen:
                    self._expired_annotations += 1
                self._pending_annotations.append(payload)
        except (KeyError, TypeError, ValueError) as exc:
            self._reject_annotation(exc)
            return
        if self._pipeline_lock.acquire(blocking=False):
            try:
                result = self._drain_annotations()
                if result is not None:
                    self._publish(result)
            finally:
                self._pipeline_lock.release()

    def _reject_annotation(self, exc):
        self._rejected_annotations += 1
        self.get_logger().warning(f"Camera labels rejected; cloud geometry retained: {exc}",
                                  throttle_duration_sec=5.0)

    def _try_apply_annotation(self, payload):
        try:
            stamp = payload["stamp"]
            transform = self._world_pose(payload["frame_id"], Time(nanoseconds=int(round(stamp*1e9))))
            camera = camera_labels_from_dict(payload, T_world_from_camera=transform,
                                              allow_yoloe=self.pipeline.config.allow_yoloe_labels)
            result = self.pipeline.annotate(camera)
        except (KeyError, TypeError, ValueError, tf2_ros.TransformException) as exc:
            self._reject_annotation(exc)
            return None
        self._applied_annotations += 1
        self._last_annotation_model = payload.get("model")
        if camera.source == "yoloe":
            self._applied_yoloe_frames.add(payload["frame_id"])
        return result

    def _publish(self, result):
        if self._last_cloud is None:
            return
        self.cloud_pub.publish(annotate_pointcloud_message(self._last_cloud, result))
        header = Header(stamp=self._last_cloud.header.stamp, frame_id=self.world_frame)
        if bool(self.get_parameter("publish_voxel_map").value):
            self.map_pub.publish(voxel_map_message(result, header))
        metadata = result_metadata(result)
        metadata.update(world_frame=self.world_frame, input_frame=self._last_cloud.header.frame_id,
                        processing_seconds=self._last_processing_seconds,
                        rejected_clouds=self._rejected_clouds,
                        rejected_annotations=self._rejected_annotations,
                        applied_annotations=self._applied_annotations,
                        expired_annotations=self._expired_annotations,
                        pending_annotations=len(self._pending_annotations),
                        last_annotation_model=self._last_annotation_model)
        self.regions_pub.publish(String(data=json.dumps(metadata)))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DenseCloudMappingNode()
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
