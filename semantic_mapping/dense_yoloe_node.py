"""Explicitly enabled, local YOLOE masks for full-cloud semantic fusion.

Inference follows calibrated live images independently of cloud publication.
The separate cloud node retains all geometry and enforces visibility.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from semantic_mapping.dense_cloud_io import annotations_payload, mask_runs, sha256_file
from semantic_mapping.ros_msgs import (
    camera_info_to_intrinsics, image_to_numpy, numpy_to_image, stamp_to_seconds,
)


def calibrated_frame(image, info, max_delta):
    """Require an already rectified image with matching optical calibration."""
    stamp = stamp_to_seconds(image.header.stamp)
    info_stamp = stamp_to_seconds(info.header.stamp)
    if stamp <= 0 or not image.header.frame_id or image.header.frame_id != info.header.frame_id:
        raise ValueError("image needs a nonzero capture stamp and matching CameraInfo optical frame")
    if info_stamp and abs(info_stamp-stamp) > max_delta:
        raise ValueError("image and CameraInfo capture times differ")
    if not np.isfinite(info.d).all() or any(abs(x) > 1e-12 for x in info.d):
        raise ValueError("YOLOE input must use a rectified image and zero-distortion CameraInfo")
    rotation = np.asarray(info.r).reshape(3, 3)
    if rotation.any() and not np.allclose(rotation, np.eye(3)):
        raise ValueError("rectified optical frame must have matching identity CameraInfo R")
    intr = camera_info_to_intrinsics(info)
    rgb = image_to_numpy(image)
    if rgb.shape != (intr.height, intr.width, 3):
        raise ValueError("image dimensions/channels do not match calibrated RGB")
    return stamp, intr, rgb


class DenseYOLOELabelsNode(Node):
    def __init__(self):
        super().__init__("dense_yoloe_labels")
        for key, value in {
            "allow_yoloe": False, "images_rectified": False,
            "checkpoint": "", "text_encoder_path": "", "device": "cuda",
            "confidence_threshold": 0.35, "prompts": ["person", "pipe", "cable", "door", "box"],
            "rgb_topic": "/camera/image_rect_color", "camera_info_topic": "/camera/camera_info",
            "annotations_topic": "/supermap/camera_annotations",
            "annotated_image_topic": "/supermap/annotated_image",
            "max_camera_time_delta": 0.20, "max_rate_hz": 20.0,
        }.items():
            self.declare_parameter(key, value)
        value = lambda name: self.get_parameter(name).value
        if not value("allow_yoloe"):
            raise ValueError("YOLOE needs the explicit allow_yoloe=true exception")
        if not value("images_rectified"):
            raise ValueError("set images_rectified=true only for calibrated rectified images")
        self.max_delta = float(value("max_camera_time_delta"))
        self.max_rate = float(value("max_rate_hz"))
        confidence = float(value("confidence_threshold"))
        if (not all(math.isfinite(x) for x in (self.max_delta, self.max_rate, confidence))
                or self.max_delta < 0 or self.max_rate <= 0
                or not 0 <= confidence <= 1):
            raise ValueError("invalid camera timing, rate or confidence configuration")
        self.prompts = list(value("prompts"))
        if not self.prompts or any(not isinstance(s, str) or not s.strip() for s in self.prompts):
            raise ValueError("prompts must contain nonempty class names")
        self.last_capture = -math.inf
        self.completed = 0
        self.detector, self.model_metadata = self._load_detector(
            value("checkpoint"), value("text_encoder_path"), value("device"), confidence)
        self.annotation_pub = self.create_publisher(String, value("annotations_topic"), 8)
        self.image_pub = self.create_publisher(Image, value("annotated_image_topic"), 2)
        self.image_sub = message_filters.Subscriber(self, Image, value("rgb_topic"), qos_profile=qos_profile_sensor_data)
        self.info_sub = message_filters.Subscriber(self, CameraInfo, value("camera_info_topic"), qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.info_sub], 10, 0.05)
        self.sync.registerCallback(self._on_camera)
        self.get_logger().info("YOLOE exception enabled: local image labels for full-cloud fusion")

    def _load_detector(self, checkpoint, encoder, device, confidence):
        # Fail before constructing Ultralytics if either local asset is missing.
        checkpoint, encoder = Path(checkpoint).expanduser(), Path(encoder).expanduser()
        if not checkpoint.is_file() or not encoder.is_file():
            raise ValueError("checkpoint and text_encoder_path must name existing local files")
        from semantic_mapping.detectors.yoloe_detector import YOLOEDetector
        detector = YOLOEDetector(str(checkpoint.resolve()), device, confidence,
                                 text_encoder_path=str(encoder.resolve()))
        # Warm up the fixed vocabulary before bag playback; no automatic downloads.
        detector.detect(np.zeros((64, 64, 3), np.uint8), self.prompts)
        metadata = {"name": "YOLOE", "exception": "explicitly_allowed_non_US_origin",
                    "checkpoint": checkpoint.name, "checkpoint_sha256": sha256_file(checkpoint),
                    "text_encoder": encoder.name, "text_encoder_sha256": sha256_file(encoder)}
        return detector, metadata

    def _on_camera(self, image, info):
        stamp = stamp_to_seconds(image.header.stamp)
        period = 1.0/self.max_rate
        if 0 <= stamp-self.last_capture < period-min(0.001, 0.01*period):
            return
        try:
            stamp, intr, rgb = calibrated_frame(image, info, self.max_delta)
            self.last_capture = stamp
            started = time.perf_counter()
            detections = self.detector.detect(rgb, self.prompts)
            payload = annotations_payload(image.header, intr, detections, self.model_metadata)
            payload["inference_seconds"] = time.perf_counter()-started
            self.annotation_pub.publish(String(data=json.dumps(payload)))
            self.completed += 1
            if self.image_pub.get_subscription_count():
                self._publish_image(rgb, image.header, detections)
            self.get_logger().info(
                f"Live camera labels #{self.completed}: {len(detections)} masks, "
                f"inference={payload['inference_seconds']:.3f}s", throttle_duration_sec=5.0)
        except (ValueError, RuntimeError) as exc:
            self.get_logger().error(f"Camera labeling failed; geometry retained: {exc}")

    def _publish_image(self, rgb, header, detections):
        import cv2
        image = rgb.copy()
        for detection in detections:
            color = np.frombuffer(hashlib.sha256(detection.label.encode()).digest()[:3], np.uint8)
            color = (color.astype(np.uint16)//2+100).astype(np.uint8)
            if detection.mask is not None:
                image[detection.mask] = (0.6*image[detection.mask]+0.4*color).astype(np.uint8)
            tint = tuple(map(int, color))
            x1, y1, x2, y2 = map(int, detection.bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), tint, 2)
            cv2.putText(image, f"{detection.label} {detection.score:.2f}", (x1, max(16, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, tint, 1, cv2.LINE_AA)
        self.image_pub.publish(numpy_to_image(image, "rgb8", header))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DenseYOLOELabelsNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
