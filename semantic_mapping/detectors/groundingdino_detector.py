"""Grounding DINO (open-set text-prompted boxes) backend.

With SAM2 mask refinement (``sam2_checkpoint`` / ``sam2_model_cfg`` through
:func:`semantic_mapping.detectors.build_detector`) this is the detector
pairing used for the instance-level segmentation results in the paper
(Sec. V-B): Grounding DINO proposes boxes for the active prompt vocabulary,
SAM2 refines each box into a per-instance mask.
"""
from __future__ import annotations

import numpy as np

from semantic_mapping.detectors.base import Detector
from semantic_mapping.types import Detection2D


class GroundingDINODetector(Detector):
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> None:
        try:
            from groundingdino.util.inference import Model as GroundingDINOModel
        except ImportError as exc:
            raise ImportError(
                "GroundingDINODetector requires the 'groundingdino' package "
                "(https://github.com/IDEA-Research/GroundingDINO)."
            ) from exc

        self.model = GroundingDINOModel(
            model_config_path=config_path, model_checkpoint_path=checkpoint_path, device=device,
        )
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def detect(self, rgb_image: np.ndarray, prompts: list[str] | None = None, **kwargs) -> list[Detection2D]:
        if not prompts:
            return []

        detections_raw = self.model.predict_with_classes(
            image=rgb_image,
            classes=prompts,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        boxes_xyxy = np.asarray(detections_raw.xyxy, dtype=np.float64)
        scores = np.asarray(detections_raw.confidence, dtype=np.float64)
        class_ids = np.asarray(detections_raw.class_id)

        detections: list[Detection2D] = []
        for i in range(boxes_xyxy.shape[0]):
            if class_ids[i] is None:
                continue
            detections.append(Detection2D(bbox=boxes_xyxy[i], label=prompts[int(class_ids[i])], score=float(scores[i])))
        return detections
