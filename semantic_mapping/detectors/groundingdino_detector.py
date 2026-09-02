"""Grounding DINO (open-set text-prompted boxes) + SAM2 (instance masks) backend.

This is the detector pairing used for the instance-level segmentation
results in the paper (Sec. V-B): Grounding DINO proposes boxes for the
active prompt vocabulary, SAM2 refines each box into a per-instance mask.
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
        sam2_checkpoint: str | None = None,
        sam2_model_cfg: str | None = None,
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

        self.sam_predictor = None
        if sam2_checkpoint and sam2_model_cfg:
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError as exc:
                raise ImportError(
                    "SAM2 refinement requires the 'sam2' package (https://github.com/facebookresearch/sam2)."
                ) from exc
            sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
            self.sam_predictor = SAM2ImagePredictor(sam2_model)

    def _segment(self, rgb_image: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray | None:
        if self.sam_predictor is None or boxes_xyxy.shape[0] == 0:
            return None
        self.sam_predictor.set_image(rgb_image)
        masks, _scores, _logits = self.sam_predictor.predict(box=boxes_xyxy, multimask_output=False)
        return masks[:, 0] if masks.ndim == 4 else masks

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

        masks = self._segment(rgb_image, boxes_xyxy)

        detections: list[Detection2D] = []
        for i in range(boxes_xyxy.shape[0]):
            if class_ids[i] is None:
                continue
            label = prompts[int(class_ids[i])]
            mask = masks[i] > 0.5 if masks is not None else None
            detections.append(Detection2D(
                bbox=boxes_xyxy[i], label=label, score=float(scores[i]), mask=mask,
            ))
        return detections
