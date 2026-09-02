"""Detector interface shared by every backend."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from semantic_mapping.types import Detection2D


class Detector(ABC):
    """A 2D open-vocabulary instance detector/segmenter.

    Implementations receive a single RGB frame and the current detection
    vocabulary (from ``config/prompts.yaml``) and return zero or more
    :class:`~semantic_mapping.types.Detection2D` instances. Detection is
    asynchronous with respect to the geometric SLAM backbone in live mode
    (Sec. IV): callers are responsible for scheduling ``detect`` at whatever
    rate their compute budget allows and feeding results back into the
    pipeline keyed by frame timestamp.
    """

    @abstractmethod
    def detect(self, rgb_image: np.ndarray, prompts: list[str] | None = None, **kwargs) -> list[Detection2D]:
        """Run detection (+ optional segmentation) on a single RGB frame.

        ``kwargs`` is an extension point for backends that need extra
        per-call context (e.g. :class:`~semantic_mapping.detectors.offline.OfflineDetector`
        needs a ``frame_id`` to look up the matching pre-baked record).
        """
        raise NotImplementedError
