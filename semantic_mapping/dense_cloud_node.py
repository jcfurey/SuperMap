"""ROS 2 full-cloud mapping, independent of camera availability and model inference."""
from __future__ import annotations

from dataclasses import fields
import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
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
        }.items():
            self.declare_parameter(name, value)
        config = DenseCloudConfig(**{item.name: self.get_parameter(item.name).value for item in fields(defaults)})
        self.pipeline = DenseCloudPipeline(config)
        self.world_frame = str(self.get_parameter("world_frame").value)
        if not self.world_frame:
            raise ValueError("world_frame must not be empty")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._last_cloud = None
        self._camera_info = {}
        self._last_processing_seconds = 0.0
        self._rejected_clouds = 0
        self._rejected_annotations = 0
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
        self.annotation_sub = (self.create_subscription(String, annotations, self._on_annotations, 10)
                               if annotations else None)
        info = self._topic("camera_info_topic")
        self.info_sub = (self.create_subscription(CameraInfo, info, self._on_camera_info, qos_profile_sensor_data)
                        if info else None)
        self.get_logger().info(
            f"Full-cloud {config.input_mode} mapping initialized; camera labels are optional, manual-only, and no models are loaded")

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
        self._last_processing_seconds = time.perf_counter()-started
        self._publish(result)

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
            if not np.isfinite(stamp) or stamp < 0:
                raise ValueError("camera stamp must be a finite, nonnegative ROS time")
            transform = self._world_pose(frame, Time(nanoseconds=int(round(stamp*1e9))))
            intrinsics = None
            if "intrinsics" not in payload:
                info = self._camera_info.get(frame)
                if info is None:
                    raise ValueError("provide intrinsics or matching CameraInfo in the annotation optical frame")
                info_stamp = stamp_to_seconds(info.header.stamp)
                if info_stamp and abs(info_stamp-stamp) > self.pipeline.config.max_camera_time_delta:
                    raise ValueError("CameraInfo is not synchronized with the camera observation")
                if any(abs(value) > 1e-12 for value in info.d):
                    raise ValueError("distorted CameraInfo cannot describe rectified masks; provide the rectified pinhole calibration")
                rotation = np.asarray(info.r).reshape(3, 3)
                if rotation.any() and not np.allclose(rotation, np.eye(3)):
                    raise ValueError("nonidentity camera rectification requires an explicit rectified optical TF and intrinsics")
                intrinsics = camera_info_to_intrinsics(info)
            camera = camera_labels_from_dict(payload, intrinsics=intrinsics, T_world_from_camera=transform)
            result = self.pipeline.annotate(camera)
        except (KeyError, TypeError, ValueError, tf2_ros.TransformException) as exc:
            self._rejected_annotations += 1
            self.get_logger().warning(f"Camera labels rejected; cloud geometry retained: {exc}")
            return
        self._publish(result)

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
                        rejected_annotations=self._rejected_annotations)
        self.regions_pub.publish(String(data=json.dumps(metadata)))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DenseCloudMappingNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
