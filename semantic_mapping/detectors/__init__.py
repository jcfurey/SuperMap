"""Pluggable open-vocabulary 2D detector backends.

SuperMap is model-agnostic (README): the same online pipeline can run
against Grounding DINO + SAM2, Ultralytics YOLOE, or pre-baked ("boxer")
detections replayed from disk. Every backend implements
:class:`semantic_mapping.detectors.base.Detector`.
"""
from semantic_mapping.detectors.base import Detector
from semantic_mapping.detectors.offline import OfflineDetector

__all__ = ["Detector", "OfflineDetector", "build_detector"]


def build_detector(name: str, **kwargs) -> Detector:
    """Factory used by the config-driven pipeline/ROS node to instantiate a detector."""
    name = name.lower()
    if name in ("offline", "boxer", "pre-baked", "prebaked"):
        return OfflineDetector(**kwargs)
    if name == "yoloe":
        from semantic_mapping.detectors.yoloe_detector import YOLOEDetector

        return YOLOEDetector(**kwargs)
    if name in ("groundingdino", "grounding_dino", "gdino"):
        from semantic_mapping.detectors.groundingdino_detector import GroundingDINODetector

        return GroundingDINODetector(**kwargs)
    raise ValueError(f"Unknown detector backend: {name!r}")
