"""Ultralytics YOLOE backend: real-time open-vocabulary detection via text prompts."""
from __future__ import annotations

import numpy as np

from semantic_mapping.detectors.base import Detector
from semantic_mapping.types import Detection2D


class YOLOEDetector(Detector):
    def __init__(
        self,
        checkpoint: str = "yoloe-v8l-seg.pt",
        device: str = "cuda",
        confidence_threshold: float = 0.25,
    ) -> None:
        try:
            from ultralytics import YOLOE
        except ImportError as exc:
            raise ImportError(
                "YOLOEDetector requires the 'ultralytics' package (pip install ultralytics)."
            ) from exc

        self.model = YOLOE(checkpoint)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._current_prompts: list[str] | None = None

    def _set_vocabulary(self, prompts: list[str]) -> None:
        if prompts == self._current_prompts:
            return
        text_embeddings = self.model.get_text_pe(prompts)
        self.model.set_classes(prompts, text_embeddings)
        self._current_prompts = list(prompts)

    def detect(self, rgb_image: np.ndarray, prompts: list[str] | None = None, **kwargs) -> list[Detection2D]:
        if prompts:
            self._set_vocabulary(prompts)

        results = self.model.predict(
            rgb_image, device=self.device, conf=self.confidence_threshold, verbose=False,
        )
        if not results:
            return []
        result = results[0]

        detections: list[Detection2D] = []
        boxes = result.boxes
        if boxes is None:
            return detections

        masks = result.masks.data.cpu().numpy() if getattr(result, "masks", None) is not None else None
        names = result.names

        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy().astype(np.float64)
            score = float(boxes.conf[i].cpu().numpy())
            class_id = int(boxes.cls[i].cpu().numpy())
            label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
            mask = masks[i] > 0.5 if masks is not None else None
            detections.append(Detection2D(bbox=bbox, label=label, score=score, mask=mask))

        return detections
