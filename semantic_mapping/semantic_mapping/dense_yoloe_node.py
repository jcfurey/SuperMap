"""Explicitly enabled, local YOLOE masks for full-cloud semantic fusion.

Inference follows calibrated live images independently of cloud publication
and publishes typed ``supermap_msgs/CameraAnnotations``. The separate cloud
node retains all geometry and enforces visibility.

A managed (lifecycle) node: the checkpoint and text encoder are loaded and
warmed up in ``on_configure``, so ``nav2_lifecycle_manager`` (or the default
``autostart``) only activates it once inference is ready.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import threading
import time

import numpy as np
from diagnostic_msgs.msg import DiagnosticStatus
import diagnostic_updater
import message_filters
from rcl_interfaces.msg import SetParametersResult
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from supermap_msgs.msg import CameraAnnotations

from semantic_mapping.dense_cloud_io import annotations_payload, annotations_to_msg, mask_runs, sha256_file  # noqa: F401
from semantic_mapping.ros_msgs import (
    camera_info_has_distortion, camera_info_to_intrinsics, image_to_numpy, numpy_to_image, rectified_camera_info,
    stamp_to_seconds,
)
from semantic_mapping.ros_node_utils import (
    AutostartLifecycleNode, TransitionCallbackReturn, declare, run_node, stable_label_color,
)

_THROTTLE = 5.0


def calibrated_frame(image, info, max_delta, *, use_projection=True):
    """Require an already rectified image with matching optical calibration.

    Returns ``(stamp, intrinsics, rgb, rectified_info)``. When the CameraInfo
    still describes the raw sensor (nonzero ``D`` or ``R != I``, as image_proc
    republishes next to ``image_rect*``), its projection matrix ``P`` describes
    the rectified pixels and ``rectified_info`` is that form; otherwise the
    frame is rejected.
    """
    stamp = stamp_to_seconds(image.header.stamp)
    info_stamp = stamp_to_seconds(info.header.stamp)
    if stamp <= 0 or not image.header.frame_id or image.header.frame_id != info.header.frame_id:
        raise ValueError("image needs a nonzero capture stamp and matching CameraInfo optical frame")
    if info_stamp and abs(info_stamp-stamp) > max_delta:
        raise ValueError("image and CameraInfo capture times differ")
    rotation = np.asarray(info.r).reshape(3, 3)
    raw_calibration = camera_info_has_distortion(info) or (rotation.any() and not np.allclose(rotation, np.eye(3)))
    if raw_calibration:
        if not use_projection:
            raise ValueError("YOLOE input must use a rectified image and zero-distortion CameraInfo")
        try:
            info = rectified_camera_info(info)
        except ValueError as exc:
            raise ValueError("CameraInfo describes a distorted/unrectified camera and has no usable P "
                             "for the rectified image") from exc
    intr = camera_info_to_intrinsics(info)
    rgb = image_to_numpy(image)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    if rgb.shape != (intr.height, intr.width, 3):
        raise ValueError("image dimensions/channels do not match calibrated RGB")
    return stamp, intr, rgb, info


class DenseYOLOELabelsNode(AutostartLifecycleNode):
    def __init__(self, **kwargs):
        super().__init__("dense_yoloe_labels", **kwargs)
        declare(self, "allow_yoloe", False, "Explicit policy exception required to run YOLOE")
        declare(self, "images_rectified", False, "Set true only for calibrated, rectified images")
        declare(self, "use_projection_matrix", True,
                "For raw CameraInfo next to a rectified image, describe masks with P (image_proc convention)")
        declare(self, "checkpoint", "", "Existing local YOLOE segmentation checkpoint")
        declare(self, "text_encoder_path", "", "Matching local MobileCLIP TorchScript encoder")
        declare(self, "device", "cuda", "Torch device")
        declare(self, "half", True, "FP16 inference on CUDA")
        declare(self, "confidence_threshold", 0.35, "Minimum detection score", read_only=False, range=(0.0, 1.0))
        declare(self, "prompts", ["person", "pipe", "cable", "door", "box"], "Fixed open vocabulary")
        declare(self, "rgb_topic", "camera/image_rect_color", "Rectified RGB image")
        declare(self, "camera_info_topic", "camera/camera_info", "CameraInfo of rgb_topic")
        declare(self, "annotations_topic", "supermap/camera_annotations", "supermap_msgs/CameraAnnotations output")
        declare(self, "annotated_image_topic", "supermap/annotated_image", "Mask overlay (only when subscribed)")
        declare(self, "max_camera_time_delta", 0.20, "Max image/CameraInfo stamp difference (s)", range=(0.0, 10.0))
        declare(self, "max_rate_hz", 20.0, "Inference rate cap by capture time", read_only=False, range=(0.01, 1000.0))
        declare(self, "sync_queue_size", 10, "Image/CameraInfo synchronizer queue", range=(1, 1000))
        declare(self, "sync_slop_s", 0.05, "Image/CameraInfo synchronizer slop (s)", range=(0.0, 1.0))
        self.detector = None
        self.model_metadata = None
        self._active = False
        self.annotation_pub = self.image_pub = None
        self.image_sub = self.info_sub = self.sync = None
        self._lock = threading.Lock()
        self._counts = {"frames": 0, "labeled": 0, "skipped_rate": 0, "failures": 0}
        self._last_inference_s = 0.0
        self._diag_previous = (dict(self._counts), time.monotonic())
        self.last_capture = -math.inf
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self._diagnostics = diagnostic_updater.Updater(self, period=1.0)
        self._diagnostics.setHardwareID(self.get_fully_qualified_name())
        self._diagnostics.add("dense yoloe labels", self._diagnose)

    @property
    def completed(self) -> int:
        with self._lock:
            return self._counts["labeled"]

    def _param(self, name):
        return self.get_parameter(name).value

    def _on_set_parameters(self, params) -> SetParametersResult:
        for param in params:
            if param.name == "confidence_threshold" and self.detector is not None:
                self.detector.confidence_threshold = float(param.value)
            elif param.name == "max_rate_hz":
                self.max_rate = float(param.value)
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------- lifecycle
    def on_configure(self, state) -> TransitionCallbackReturn:
        try:
            self._configure()
        except Exception as exc:  # noqa: BLE001 - model loading errors fail the transition, not the process
            self.get_logger().error(f"configure failed ({type(exc).__name__}): {exc}")
            self._teardown()
            return TransitionCallbackReturn.FAILURE
        return TransitionCallbackReturn.SUCCESS

    def _configure(self) -> None:
        if not self._param("allow_yoloe"):
            raise ValueError("YOLOE needs the explicit allow_yoloe=true exception")
        if not self._param("images_rectified"):
            raise ValueError("set images_rectified=true only for calibrated rectified images")
        self.max_delta = float(self._param("max_camera_time_delta"))
        self.max_rate = float(self._param("max_rate_hz"))
        confidence = float(self._param("confidence_threshold"))
        if (not all(math.isfinite(x) for x in (self.max_delta, self.max_rate, confidence))
                or self.max_delta < 0 or self.max_rate <= 0 or not 0 <= confidence <= 1):
            raise ValueError("invalid camera timing, rate or confidence configuration")
        self.prompts = list(self._param("prompts"))
        if not self.prompts or any(not isinstance(s, str) or not s.strip() for s in self.prompts):
            raise ValueError("prompts must contain nonempty class names")
        self.use_projection = bool(self._param("use_projection_matrix"))
        self.detector, self.model_metadata = self._load_detector(
            self._param("checkpoint"), self._param("text_encoder_path"), self._param("device"), confidence)
        self._model_text = json.dumps(self.model_metadata, sort_keys=True)
        self.annotation_pub = self.create_lifecycle_publisher(CameraAnnotations, self._param("annotations_topic"), 8)
        self.image_pub = self.create_lifecycle_publisher(Image, self._param("annotated_image_topic"), 2)
        self.image_sub = message_filters.Subscriber(self, Image, self._param("rgb_topic"),
                                                    qos_profile=qos_profile_sensor_data)
        self.info_sub = message_filters.Subscriber(self, CameraInfo, self._param("camera_info_topic"),
                                                   qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.info_sub], int(self._param("sync_queue_size")), float(self._param("sync_slop_s")))
        self.sync.registerCallback(self._on_camera)
        self.last_capture = -math.inf
        self.get_logger().info("YOLOE exception enabled: local image labels for full-cloud fusion")

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
        for name in ("image_sub", "info_sub"):
            sub = getattr(self, name)
            if sub is not None:
                self.destroy_subscription(sub.sub)
                setattr(self, name, None)
        self.sync = None
        for name in ("annotation_pub", "image_pub"):
            pub = getattr(self, name)
            if pub is not None:
                self.destroy_lifecycle_publisher(pub)
                setattr(self, name, None)
        self.detector = None  # releases the model (and its GPU memory) with the last reference

    def destroy_node(self):
        self._teardown()
        return super().destroy_node()

    def _load_detector(self, checkpoint, encoder, device, confidence):
        # Fail before constructing Ultralytics if either local asset is missing.
        checkpoint, encoder = Path(checkpoint).expanduser(), Path(encoder).expanduser()
        if not checkpoint.is_file() or not encoder.is_file():
            raise ValueError("checkpoint and text_encoder_path must name existing local files")
        from semantic_mapping.detectors.yoloe_detector import YOLOEDetector
        detector = YOLOEDetector(str(checkpoint.resolve()), device, confidence,
                                 text_encoder_path=str(encoder.resolve()), half=bool(self._param("half")))
        # Warm up the fixed vocabulary before bag playback; no automatic downloads.
        detector.detect(np.zeros((64, 64, 3), np.uint8), self.prompts)
        metadata = {"name": "YOLOE", "exception": "explicitly_allowed_non_US_origin",
                    "checkpoint": checkpoint.name, "checkpoint_sha256": sha256_file(checkpoint),
                    "text_encoder": encoder.name, "text_encoder_sha256": sha256_file(encoder)}
        return detector, metadata

    # ------------------------------------------------------------- inference
    def _on_camera(self, image, info):
        if not self._active or self.detector is None:
            return
        with self._lock:
            self._counts["frames"] += 1
        stamp = stamp_to_seconds(image.header.stamp)
        period = 1.0/self.max_rate
        if 0 <= stamp-self.last_capture < period-min(0.001, 0.01*period):
            with self._lock:
                self._counts["skipped_rate"] += 1
            return
        try:
            stamp, intr, rgb, rectified_info = calibrated_frame(image, info, self.max_delta,
                                                                use_projection=self.use_projection)
            self.last_capture = stamp
            started = time.perf_counter()
            detections = self.detector.detect(rgb, self.prompts)
            self._last_inference_s = time.perf_counter()-started
            message = annotations_to_msg(image.header, rectified_info, detections, self._model_text, source="yoloe")
            self.annotation_pub.publish(message)
            with self._lock:
                self._counts["labeled"] += 1
                completed = self._counts["labeled"]
            if self.image_pub.get_subscription_count():
                self._publish_image(rgb, image.header, detections)
            self.get_logger().info(
                f"Live camera labels #{completed}: {len(detections)} masks, "
                f"inference={self._last_inference_s:.3f}s", throttle_duration_sec=_THROTTLE)
        except Exception as exc:  # noqa: BLE001 - cv2.error / Ultralytics / CUDA errors must not kill the node
            with self._lock:
                self._counts["failures"] += 1
            self.get_logger().error(f"Camera labeling failed; geometry retained ({type(exc).__name__}): {exc}",
                                    throttle_duration_sec=_THROTTLE)

    def _publish_image(self, rgb, header, detections):
        import cv2
        image = rgb.copy()
        for detection in detections:
            tint = tuple(int(round(c*255)) for c in stable_label_color(detection.label))
            if detection.mask is not None:
                pixels = image[detection.mask]
                image[detection.mask] = (0.6*pixels+0.4*np.asarray(tint)).astype(np.uint8)
            x1, y1, x2, y2 = map(int, detection.bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), tint, 2)
            cv2.putText(image, f"{detection.label} {detection.score:.2f}", (x1, max(16, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, tint, 1, cv2.LINE_AA)
        self.image_pub.publish(numpy_to_image(image, "rgb8", header))

    def _diagnose(self, stat):
        now = time.monotonic()
        with self._lock:
            counts = dict(self._counts)
        previous, since = self._diag_previous
        self._diag_previous = (counts, now)
        elapsed = max(now-since, 1e-9)
        rate = (counts["labeled"]-previous.get("labeled", 0))/elapsed
        failures = counts["failures"]-previous.get("failures", 0)
        if self.detector is None:
            stat.summary(DiagnosticStatus.OK, "unconfigured")
        elif not self._active:
            stat.summary(DiagnosticStatus.OK, "inactive")
        elif failures:
            stat.summary(DiagnosticStatus.WARN, f"{failures} labeling failures since last report")
        else:
            stat.summary(DiagnosticStatus.OK, "ok")
        stat.add("label_rate_hz", f"{rate:.2f}")
        stat.add("last_inference_s", f"{self._last_inference_s:.3f}")
        for key, value in counts.items():
            stat.add(key, str(value))
        return stat


def main(args=None):
    run_node(DenseYOLOELabelsNode, args)


if __name__ == "__main__":
    main()
