"""Exercise the Ultralytics adapter contract without Torch or model downloads."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from semantic_mapping.detectors.yoloe_detector import YOLOEDetector
from semantic_mapping.pipeline import SemanticMappingPipeline
from semantic_mapping.types import CameraIntrinsics, Observation, StampedPose


class _Tensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    def __getitem__(self, index):
        return _Tensor(self.data[index])

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class _Boxes:
    xyxy = _Tensor([[200.0, 100.0, 240.0, 140.0]])
    conf = _Tensor([0.9])
    cls = _Tensor([0])

    def __len__(self):
        return 1


def _result(shape):
    masks = np.zeros((1, *shape), dtype=np.float32)
    masks[:, 100:140, 200:240] = 1.0
    return SimpleNamespace(boxes=_Boxes(), names={0: 'chair'}, masks=SimpleNamespace(data=_Tensor(masks)))


def test_yoloe_converts_rgb_and_requests_masks_that_fuse_at_camera_resolution(monkeypatch):
    captured = {}

    def predict(image, **kwargs):
        captured.update(image=image.copy(), **kwargs)
        # The default Ultralytics output follows the inference resolution.
        return [_result(image.shape[:2] if kwargs.get('retina_masks') else (384, 640))]

    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLOE=lambda path: SimpleNamespace(predict=predict)))
    detector = YOLOEDetector(device='cpu')
    rgb = np.full((720, 1280, 3), [230, 60, 10], dtype=np.uint8)
    detections = detector.detect(rgb)
    np.testing.assert_array_equal(captured['image'][0, 0], [10, 60, 230])
    np.testing.assert_array_equal(rgb[0, 0], [230, 60, 10])
    assert captured['image'].flags.c_contiguous
    assert detections[0].mask.shape == (720, 1280)
    assert detections[0].mask.dtype == np.bool_
    result = SemanticMappingPipeline().process_frame(Observation(
        stamp=0.0, pose=StampedPose(0.0, np.eye(4)),
        intrinsics=CameraIntrinsics(1000, 1000, 640, 360, 1280, 720),
        rgb=rgb, depth=np.full((720, 1280), 2.0), detections=detections))
    assert len(result.objects) == 1 and len(result.objects[0].points_world) > 0


def test_yoloe_rejects_a_backend_that_returns_inference_sized_masks(monkeypatch):
    model = SimpleNamespace(predict=lambda *args, **kwargs: [_result((384, 640))])
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLOE=lambda path: model))
    with pytest.raises(ValueError, match='boolean mask of shape'):
        YOLOEDetector(device='cpu').detect(np.zeros((720, 1280, 3), dtype=np.uint8))
