"""SAM2 mask refinement for any box detector.

The paper pairs Grounding DINO boxes with SAM2 instance masks (Sec. V-B),
and masks matter well beyond that pairing: a box always carries some
background, which the depth-consistency filter and per-point membership
pruning can only partly reject, while a mask keeps it out of the object's
3D point set from the start (on the synthetic scene, masks lift detection
recall from 0.47 to 0.97). This wrapper adds a SAM2 mask to every box a
wrapped detector returns, so YOLOE's detection-only checkpoints, pre-baked
box records, and Grounding DINO all get the same treatment.
"""
from __future__ import annotations

import numpy as np

from semantic_mapping.detectors.base import Detector
from semantic_mapping.types import Detection2D


def build_sam2_predictor(checkpoint: str, model_cfg: str, device: str = "cuda"):
    """Instantiate a SAM2 image predictor (lazy import: SAM2 is an optional dependency)."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise ImportError(
            "SAM2 mask refinement requires the 'sam2' package (https://github.com/facebookresearch/sam2)."
        ) from exc
    return SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))


class SAM2MaskRefiner(Detector):
    """Run a box detector, then segment each box with SAM2.

    ``predictor`` must offer SAM2's image-predictor interface: ``set_image``
    and ``predict(box=..., multimask_output=False)`` returning masks scored
    per box. Boxes that already carry a mask are left alone unless
    ``refine_existing_masks`` is set.
    """

    def __init__(
        self,
        base: Detector,
        predictor=None,
        sam2_checkpoint: str | None = None,
        sam2_model_cfg: str | None = None,
        device: str = "cuda",
        mask_threshold: float = 0.5,
        refine_existing_masks: bool = False,
    ) -> None:
        if predictor is None:
            if not (sam2_checkpoint and sam2_model_cfg):
                raise ValueError("SAM2MaskRefiner needs a predictor or both sam2_checkpoint and sam2_model_cfg")
            predictor = build_sam2_predictor(sam2_checkpoint, sam2_model_cfg, device)
        self.base = base
        self.predictor = predictor
        self.mask_threshold = mask_threshold
        self.refine_existing_masks = refine_existing_masks

    def segment(self, rgb_image: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        """(N, H, W) boolean masks for N boxes."""
        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
        if boxes_xyxy.shape[0] == 0:
            return np.zeros((0,) + rgb_image.shape[:2], dtype=bool)
        self.predictor.set_image(rgb_image)
        masks, _scores, _logits = self.predictor.predict(box=boxes_xyxy, multimask_output=False)
        masks = np.asarray(masks)
        if masks.ndim == 4:  # (N, 1, H, W) for batched boxes
            masks = masks[:, 0]
        elif masks.ndim == 2:  # a single box may come back unbatched
            masks = masks[None]
        return masks > self.mask_threshold

    def detect(self, rgb_image: np.ndarray, prompts: list[str] | None = None, **kwargs) -> list[Detection2D]:
        detections = self.base.detect(rgb_image, prompts=prompts, **kwargs)
        targets = [d for d in detections if d.mask is None or self.refine_existing_masks]
        if not targets:
            return detections
        masks = self.segment(rgb_image, np.stack([d.bbox for d in targets]))
        for detection, mask in zip(targets, masks):
            detection.mask = mask
        return detections
