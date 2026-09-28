"""Grounding DINO's image contract, without model weights or inference."""
import sys
from types import SimpleNamespace

import numpy as np

from semantic_mapping.detectors.groundingdino_detector import GroundingDINODetector


def test_groundingdino_converts_rgb_to_contiguous_bgr(monkeypatch):
    seen = []

    def predict(image, **kwargs):
        seen.append(image)
        assert kwargs['classes'] == ['chair']
        return SimpleNamespace(xyxy=[[1., 2., 3., 4.]], confidence=[.9], class_id=[0])

    monkeypatch.setitem(sys.modules, 'groundingdino.util.inference', SimpleNamespace(
        Model=lambda **kwargs: SimpleNamespace(predict_with_classes=predict)))
    detector = GroundingDINODetector('config', 'checkpoint', device='cpu')
    rgb = np.full((4, 5, 3), [230, 60, 10], dtype=np.uint8)
    detections = detector.detect(rgb, ['chair'])
    np.testing.assert_array_equal(seen[0][0, 0], [10, 60, 230])
    np.testing.assert_array_equal(rgb[0, 0], [230, 60, 10])
    assert seen[0].flags.c_contiguous
    assert detections[0].label == 'chair' and detections[0].score == .9
    np.testing.assert_array_equal(detections[0].bbox, [1., 2., 3., 4.])
