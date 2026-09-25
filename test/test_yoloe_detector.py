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


def test_explicit_local_text_encoder_is_used_once_for_fixed_vocabulary(monkeypatch, tmp_path):
    calls = []
    encoder = tmp_path/'encoder.ts'
    encoder.write_bytes(b'fake')
    inner = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device='cpu')]),
                            get_text_pe=lambda prompts, **kwargs: calls.append(('embeddings', kwargs)))
    model = SimpleNamespace(model=inner, set_classes=lambda prompts, pe: calls.append(('classes', prompts)))
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLOE=lambda path: model))
    monkeypatch.setitem(sys.modules, 'ultralytics.nn.text_model', SimpleNamespace(
        MobileCLIPTS=lambda **kwargs: calls.append(('encoder', kwargs))))
    detector = YOLOEDetector(device='cpu', text_encoder_path=str(encoder))
    detector._set_vocabulary(['pipe'])
    detector._set_vocabulary(['pipe'])
    assert calls == [('encoder', {'device': 'cpu', 'weight': str(encoder)}),
                     ('embeddings', {'cache_clip_model': True}), ('classes', ['pipe'])]


def test_missing_explicit_encoder_fails_before_model_construction(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLOE=lambda path: calls.append(path)))
    with pytest.raises(ValueError, match='existing local'):
        YOLOEDetector(text_encoder_path=str(tmp_path/'missing.ts'))
    assert not calls
