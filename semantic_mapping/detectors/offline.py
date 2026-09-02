"""Replays pre-baked ("boxer") 2D detections from disk instead of running a model.

Useful for offline evaluation/ablation against a fixed detection set, and as
a dependency-free backend for CI and unit tests. Each frame's detections are
stored as a small JSON record:

    {
      "detections": [
        {"bbox": [x1, y1, x2, y2], "label": "chair", "score": 0.91, "mask": "0001_0.png"},
        ...
      ]
    }

``mask`` is an optional path (relative to the same directory) to a binary
PNG instance mask.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from semantic_mapping.detectors.base import Detector
from semantic_mapping.types import Detection2D


class OfflineDetector(Detector):
    def __init__(self, detections_dir: str | Path, frame_id_pattern: str = "{frame_id:06d}.json") -> None:
        self.detections_dir = Path(detections_dir)
        self.frame_id_pattern = frame_id_pattern

    def _load_mask(self, mask_relpath: str) -> np.ndarray | None:
        mask_path = self.detections_dir / mask_relpath
        if not mask_path.exists():
            return None
        try:
            import cv2

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        except ImportError:
            from PIL import Image

            mask = np.array(Image.open(mask_path).convert("L"))
        return mask > 0 if mask is not None else None

    def detect(self, rgb_image: np.ndarray, prompts: list[str] | None = None, **kwargs) -> list[Detection2D]:
        frame_id = kwargs.get("frame_id")
        if frame_id is None:
            raise ValueError("OfflineDetector.detect requires a frame_id=... keyword argument")

        record_path = self.detections_dir / self.frame_id_pattern.format(frame_id=frame_id)
        if not record_path.exists():
            return []

        with open(record_path) as f:
            record = json.load(f)

        allowed = set(prompts) if prompts else None
        detections: list[Detection2D] = []
        for entry in record.get("detections", []):
            label = entry["label"]
            if allowed is not None and label not in allowed:
                continue
            mask = self._load_mask(entry["mask"]) if entry.get("mask") else None
            detections.append(Detection2D(
                bbox=np.array(entry["bbox"], dtype=np.float64),
                label=label,
                score=float(entry.get("score", 1.0)),
                mask=mask,
            ))
        return detections
