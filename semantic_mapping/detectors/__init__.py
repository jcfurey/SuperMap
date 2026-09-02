"""Pluggable open-vocabulary 2D detector backends.

SuperMap is model-agnostic (README): the same online pipeline can run
against Grounding DINO + SAM2, Ultralytics YOLOE, or pre-baked ("boxer")
detections replayed from disk. Every backend implements
:class:`semantic_mapping.detectors.base.Detector`, and any of them can be
wrapped with SAM2 mask refinement (``sam2_checkpoint`` + ``sam2_model_cfg``).
"""
from semantic_mapping.detectors.base import Detector
from semantic_mapping.detectors.offline import OfflineDetector

__all__ = ["Detector", "OfflineDetector", "build_detector"]

_SAM2_KEYS = ("sam2_checkpoint", "sam2_model_cfg", "sam2_predictor", "sam2_mask_threshold", "sam2_refine_existing_masks")


def build_detector(name: str, **kwargs) -> Detector:
    """Factory used by the config-driven pipeline/ROS node to instantiate a detector.

    ``sam2_*`` keyword arguments apply to every backend: when
    ``sam2_checkpoint`` and ``sam2_model_cfg`` (or an injected
    ``sam2_predictor``) are given, the detector is wrapped in
    :class:`~semantic_mapping.detectors.sam2_refiner.SAM2MaskRefiner`.
    """
    name = name.lower()
    sam2 = {key: kwargs.pop(key) for key in _SAM2_KEYS if key in kwargs}
    if name in ("offline", "boxer", "pre-baked", "prebaked"):
        detector: Detector = OfflineDetector(**kwargs)
    elif name == "yoloe":
        from semantic_mapping.detectors.yoloe_detector import YOLOEDetector

        detector = YOLOEDetector(**kwargs)
    elif name in ("groundingdino", "grounding_dino", "gdino"):
        from semantic_mapping.detectors.groundingdino_detector import GroundingDINODetector

        detector = GroundingDINODetector(**kwargs)
    else:
        raise ValueError(f"Unknown detector backend: {name!r}")

    if sam2.get("sam2_predictor") is not None or (sam2.get("sam2_checkpoint") and sam2.get("sam2_model_cfg")):
        from semantic_mapping.detectors.sam2_refiner import SAM2MaskRefiner

        return SAM2MaskRefiner(
            detector,
            predictor=sam2.get("sam2_predictor"),
            sam2_checkpoint=sam2.get("sam2_checkpoint"),
            sam2_model_cfg=sam2.get("sam2_model_cfg"),
            device=str(kwargs.get("device", "cuda")),
            mask_threshold=float(sam2.get("sam2_mask_threshold", 0.5)),
            refine_existing_masks=bool(sam2.get("sam2_refine_existing_masks", False)),
        )
    return detector
