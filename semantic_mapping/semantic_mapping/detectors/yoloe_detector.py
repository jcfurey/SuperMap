"""Ultralytics YOLOE backend: real-time open-vocabulary detection via text prompts."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from semantic_mapping.detectors.base import Detector
from semantic_mapping.types import Detection2D


class YOLOEDetector(Detector):
    def __init__(
        self,
        checkpoint: str = "yoloe-v8l-seg.pt",
        device: str = "cuda",
        confidence_threshold: float = 0.25,
        text_encoder_path: str | None = None,
        half: bool = True,
        mask_threshold: float = 0.5,
    ) -> None:
        if text_encoder_path is not None and not Path(text_encoder_path).is_file():
            raise ValueError("text_encoder_path must name an existing local TorchScript encoder")
        try:
            from ultralytics import YOLOE
        except ImportError as exc:
            raise ImportError(
                "YOLOEDetector requires the 'ultralytics' package (pip install ultralytics)."
            ) from exc

        self.model = YOLOE(checkpoint)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.text_encoder_path = text_encoder_path
        # FP16 inference only on CUDA; CPU half precision is slow or unsupported.
        self.half = bool(half) and str(device).startswith("cuda")
        self.mask_threshold = float(mask_threshold)
        self._current_prompts: list[str] | None = None

    def _set_vocabulary(self, prompts: list[str]) -> None:
        if prompts == self._current_prompts:
            return
        if self.text_encoder_path is None:
            text_embeddings = self.model.get_text_pe(prompts)
        else:
            # Explicit local encoder: do not fetch an implicit model by name.
            from ultralytics.nn.text_model import MobileCLIPTS
            device = next(self.model.model.parameters()).device
            self.model.model.clip_model = MobileCLIPTS(device=device, weight=self.text_encoder_path)
            text_embeddings = self.model.model.get_text_pe(prompts, cache_clip_model=True)
        self.model.set_classes(prompts, text_embeddings)
        self._current_prompts = list(prompts)

    def detect(self, rgb_image: np.ndarray, prompts: list[str] | None = None, **kwargs) -> list[Detection2D]:
        if prompts:
            self._set_vocabulary(prompts)

        predict_kwargs = dict(device=self.device, conf=self.confidence_threshold, verbose=False, retina_masks=True)
        if self.half:
            predict_kwargs["half"] = True
        results = self.model.predict(
            np.ascontiguousarray(rgb_image[:, :, ::-1]),  # Ultralytics numpy inputs are BGR.
            **predict_kwargs,
        )
        if not results:
            return []
        result = results[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        # One device->host transfer for all boxes (x1, y1, x2, y2, conf, cls[, id]) instead of three per box.
        packed = getattr(boxes, "data", None)
        if packed is not None:
            table = _to_numpy(packed)
            xyxy, conf, cls = table[:, :4], table[:, -2], table[:, -1]
        else:
            xyxy, conf, cls = _to_numpy(boxes.xyxy), _to_numpy(boxes.conf), _to_numpy(boxes.cls)
        # Threshold on the device and transfer bool masks (1 byte/pixel, not 4).
        masks = None
        if getattr(result, "masks", None) is not None:
            masks = _to_numpy(result.masks.data > self.mask_threshold).astype(bool, copy=False)
        names = result.names

        detections: list[Detection2D] = []
        for i in range(len(xyxy)):
            class_id = int(cls[i])
            label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
            mask = masks[i] if masks is not None else None
            detection = Detection2D(bbox=np.asarray(xyxy[i], dtype=np.float64), label=label,
                                    score=float(conf[i]), mask=mask)
            detection.validate_mask(rgb_image.shape[:2])
            detections.append(detection)

        return detections


def _to_numpy(tensor) -> np.ndarray:
    """Host copy of a torch tensor (or an array-like), in one transfer."""
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    return np.asarray(tensor)
