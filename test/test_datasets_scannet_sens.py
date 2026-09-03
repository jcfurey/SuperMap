import json
import struct
import zlib

import numpy as np
import pytest

from semantic_mapping import datasets_scannet as sn
from semantic_mapping.datasets import load_dataset
from test.test_datasets_scannet import _write_ply

cv2 = pytest.importorskip("cv2")


def _write_sens(path, frames, color_size=(32, 24), depth_size=(16, 12), depth_shift=1000.0):
    """Write a .sens file in ScanNet's layout: header, then per-frame pose,
    stamps, sizes, JPEG colour and zlib-compressed 16-bit depth."""
    K_color = np.array([[20.0, 0, 16.0, 0], [0, 20.0, 12.0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    K_depth = np.array([[10.0, 0, 8.0, 0], [0, 10.0, 6.0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    with open(path, "wb") as f:
        f.write(struct.pack("<I", 4))
        name = b"StructureSensor"
        f.write(struct.pack("<Q", len(name)) + name)
        for matrix in (K_color, np.eye(4), K_depth, np.eye(4)):
            f.write(struct.pack("<16f", *matrix.ravel()))
        f.write(struct.pack("<ii", 2, 1))  # jpeg colour, zlib ushort depth
        f.write(struct.pack("<4I", color_size[0], color_size[1], depth_size[0], depth_size[1]))
        f.write(struct.pack("<f", depth_shift))
        f.write(struct.pack("<Q", len(frames)))
        for pose, stamp_us, rgb, depth_mm in frames:
            ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            assert ok
            depth_bytes = zlib.compress(depth_mm.astype("<u2").tobytes())
            f.write(struct.pack("<16f", *np.asarray(pose, dtype=np.float32).ravel()))
            f.write(struct.pack("<QQQQ", stamp_us, stamp_us, len(jpeg), len(depth_bytes)))
            f.write(jpeg.tobytes())
            f.write(depth_bytes)


def _scene(tmp_path):
    scene = tmp_path / "scene0010_00"
    scene.mkdir()
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    rgb[:, :16] = (200, 30, 30)
    depth = np.full((12, 16), 2500, dtype=np.uint16)
    depth[0, 0] = 0
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    bad = np.full((4, 4), -np.inf)
    frames = [(pose, 1_000_000, rgb, depth), (bad, 1_033_333, rgb, depth), (pose, 1_066_666, rgb, depth)]
    _write_sens(scene / "scene0010_00.sens", frames)
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    _write_ply(scene / "scene0010_00_vh_clean_2.labels.ply", xyz, np.array([5, 1]))
    (scene / "scene0010_00_vh_clean_2.0.010000.segs.json").write_text(json.dumps({"segIndices": [3, 4]}))
    (scene / "scene0010_00_vh_clean.aggregation.json").write_text(json.dumps({
        "segGroups": [{"id": 0, "objectId": 0, "label": "chair", "segments": [3]}]}))
    return scene


def test_sens_header_frames_and_decoding(tmp_path):
    scene = _scene(tmp_path)
    seq = sn.ScanNetSensSequence(scene / "scene0010_00.sens")
    assert seq.sensor_name == "StructureSensor" and seq.num_frames == 3
    assert seq.color_compression == "jpeg" and seq.depth_compression == "zlib_ushort"
    assert (seq.intrinsics.width, seq.intrinsics.height, seq.intrinsics.fx, seq.intrinsics.cx) == (16, 12, 10.0, 8.0)
    assert seq.frame_ids == [0, 2]  # the -inf pose is skipped

    frames = list(seq)
    assert frames[0].rgb.shape == (12, 16, 3) and frames[0].rgb[6, 3, 0] > 150 and frames[0].rgb[6, 12, 0] < 60
    assert frames[0].depth.dtype == np.float32 and frames[0].depth[5, 5] == pytest.approx(2.5)
    assert frames[0].depth[0, 0] == 0.0
    assert np.allclose(frames[1].T_world_from_cam[:3, 3], [1.0, 2.0, 3.0])
    assert frames[1].stamp == pytest.approx(1.066666)
    assert seq.observation(frames[0], []).intrinsics.fy == 10.0

    gt = seq.ground_truth_points()
    assert list(gt.labels) == ["chair", "wall"] and list(gt.instance_ids) == [0, -1]


def test_sens_frame_skip_and_dataset_detection(tmp_path):
    scene = _scene(tmp_path)
    assert sn.ScanNetSensSequence(scene / "scene0010_00.sens", frame_skip=2).frame_ids == [0]
    assert sn.ScanNetSensSequence(scene / "scene0010_00.sens", max_frames=1).frame_ids == [0]
    assert isinstance(load_dataset(scene), sn.ScanNetSensSequence)               # directory holding one .sens
    assert isinstance(load_dataset(scene / "scene0010_00.sens"), sn.ScanNetSensSequence)
    assert sn.find_sens(tmp_path) is None


def test_sens_rejects_unsupported_depth_compression(tmp_path):
    scene = _scene(tmp_path)
    path = scene / "scene0010_00.sens"
    data = bytearray(path.read_bytes())
    # depth compression field follows: version(4) + strlen(8) + name(15) + 4 matrices(256) + colour compression(4)
    offset = 4 + 8 + len(b"StructureSensor") + 256 + 4
    data[offset:offset + 4] = struct.pack("<i", 2)  # occi_ushort
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="depth compression"):
        sn.ScanNetSensSequence(path)
