import json

import numpy as np
import pytest

from semantic_mapping.detectors import build_detector
from semantic_mapping.detectors.base import Detector
from semantic_mapping.detectors.sam2_refiner import SAM2MaskRefiner
from semantic_mapping.types import Detection2D


class FakePredictor:
    """Mimics SAM2ImagePredictor: fills each box with a mask, batched as (N, 1, H, W)."""

    def __init__(self):
        self.images = []
        self.boxes = None

    def set_image(self, image):
        self.images.append(image)

    def predict(self, box, multimask_output=False):
        self.boxes = np.asarray(box)
        h, w = self.images[-1].shape[:2]
        masks = np.zeros((self.boxes.shape[0], 1, h, w), dtype=np.float32)
        for i, (x1, y1, x2, y2) in enumerate(self.boxes.astype(int)):
            masks[i, 0, y1:y2, x1:x2] = 0.9
        return masks, np.full(self.boxes.shape[0], 0.8), None


class BoxDetector(Detector):
    def __init__(self, detections):
        self.detections = detections

    def detect(self, rgb_image, prompts=None, **kwargs):
        return list(self.detections)


def test_refiner_adds_masks_to_box_only_detections_and_keeps_existing_ones():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    existing = np.zeros((20, 30), dtype=bool)
    existing[0, 0] = True
    base = BoxDetector([
        Detection2D(bbox=np.array([2.0, 3.0, 8.0, 9.0]), label="chair", score=0.9),
        Detection2D(bbox=np.array([10.0, 10.0, 20.0, 18.0]), label="table", score=0.8, mask=existing),
    ])
    predictor = FakePredictor()
    refiner = SAM2MaskRefiner(base, predictor=predictor)

    detections = refiner.detect(image)
    assert len(detections) == 2 and len(predictor.images) == 1
    assert predictor.boxes.shape == (1, 4)  # only the mask-less box was segmented
    chair = detections[0].mask
    assert chair.dtype == bool and chair.shape == (20, 30)
    assert chair[3:9, 2:8].all() and chair.sum() == 36
    assert detections[1].mask is existing

    refine_all = SAM2MaskRefiner(base, predictor=FakePredictor(), refine_existing_masks=True)
    detections = refine_all.detect(image)
    assert detections[1].mask.sum() == 80  # replaced by the 10x8 box mask


def test_refiner_skips_predictor_when_nothing_to_segment():
    predictor = FakePredictor()
    refiner = SAM2MaskRefiner(BoxDetector([]), predictor=predictor)
    assert refiner.detect(np.zeros((4, 4, 3), dtype=np.uint8)) == []
    assert predictor.images == []


def test_refiner_requires_predictor_or_checkpoints():
    with pytest.raises(ValueError):
        SAM2MaskRefiner(BoxDetector([]))


def test_factory_wraps_any_backend(tmp_path):
    (tmp_path / "000000.json").write_text(json.dumps({"detections": [
        {"bbox": [1, 1, 5, 6], "label": "cup", "score": 0.7},
    ]}))
    plain = build_detector("offline", detections_dir=tmp_path)
    assert not isinstance(plain, SAM2MaskRefiner)

    wrapped = build_detector("offline", detections_dir=tmp_path, sam2_predictor=FakePredictor(),
                             sam2_mask_threshold=0.5)
    assert isinstance(wrapped, SAM2MaskRefiner)
    detections = wrapped.detect(np.zeros((8, 8, 3), dtype=np.uint8), frame_id=0)
    assert detections[0].label == "cup" and detections[0].mask.sum() == 20

    # An incomplete SAM2 spec (checkpoint without config) means no refinement, not an error.
    assert not isinstance(build_detector("offline", detections_dir=tmp_path, sam2_checkpoint="ckpt.pt"), SAM2MaskRefiner)
