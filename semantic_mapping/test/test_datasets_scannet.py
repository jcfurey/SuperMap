import json

import numpy as np
import pytest

from semantic_mapping import datasets_scannet as sn
from semantic_mapping.datasets import load_dataset

cv2 = pytest.importorskip("cv2")


def _write_ply(path, xyz: np.ndarray, labels: np.ndarray, ascii_format: bool = False) -> None:
    n = xyz.shape[0]
    fmt = "ascii" if ascii_format else "binary_little_endian"
    header = (
        f"ply\nformat {fmt} 1.0\ncomment made by a test\nelement vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n"
        "property ushort label\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n"
    )
    if ascii_format:
        rows = "\n".join(f"{x} {y} {z} 1 2 3 255 {int(l)}" for (x, y, z), l in zip(xyz, labels))
        path.write_text(header + rows + "\n3 0 1 2\n")
        return
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"),
                      ("blue", "u1"), ("alpha", "u1"), ("label", "<u2")])
    vertices = np.zeros(n, dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertices["label"] = labels
    face = np.array([3], dtype=np.uint8).tobytes() + np.array([0, 1, 2], dtype="<i4").tobytes()
    path.write_bytes(header.encode("ascii") + vertices.tobytes() + face)


def _write_scene(root, ascii_ply: bool = False):
    scene = root / "scene0000_00"
    for sub in ("color", "depth", "pose", "intrinsic"):
        (scene / sub).mkdir(parents=True)
    K = np.array([[10.0, 0, 8.0], [0, 10.0, 6.0], [0, 0, 1.0]])
    np.savetxt(scene / "intrinsic" / "intrinsic_depth.txt", np.block([[K, np.zeros((3, 1))], [np.zeros((1, 3)), 1]]))

    depth_mm = np.full((12, 16), 2000, dtype=np.uint16)  # 2 m
    depth_mm[0, 0] = 0
    for frame_id, pose in ((0, np.eye(4)), (5, np.full((4, 4), -np.inf)), (10, np.eye(4))):
        cv2.imwrite(str(scene / "depth" / f"{frame_id}.png"), depth_mm)
        cv2.imwrite(str(scene / "color" / f"{frame_id}.jpg"), np.full((24, 32, 3), 120, dtype=np.uint8))
        np.savetxt(scene / "pose" / f"{frame_id}.txt", pose)

    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    labels = np.array([5, 5, 1, 0])  # chair, chair, wall, unlabeled
    _write_ply(scene / "scene0000_00_vh_clean_2.labels.ply", xyz, labels, ascii_format=ascii_ply)
    (scene / "scene0000_00_vh_clean_2.0.010000.segs.json").write_text(json.dumps({"segIndices": [7, 7, 9, 11]}))
    (scene / "scene0000_00_vh_clean.aggregation.json").write_text(json.dumps({
        "segGroups": [{"id": 0, "objectId": 0, "label": "chair", "segments": [7]},
                      {"id": 1, "objectId": 1, "label": "wall", "segments": [9]}],
    }))
    return scene


def test_scene_frames_skip_invalid_poses_and_align_rgb_to_depth(tmp_path):
    scene = sn.ScanNetScene(_write_scene(tmp_path))
    assert scene.frame_ids == [0, 10]
    assert (scene.intrinsics.width, scene.intrinsics.height) == (16, 12)
    frames = list(scene)
    assert frames[0].rgb.shape == (12, 16, 3)
    assert frames[0].depth.dtype == np.float32 and frames[0].depth[5, 5] == pytest.approx(2.0)
    assert frames[0].depth[0, 0] == 0.0
    assert frames[1].stamp == pytest.approx(10 / sn.SCANNET_FPS)
    assert np.allclose(frames[1].T_world_from_cam, np.eye(4))
    observation = scene.observation(frames[0], [])
    assert observation.intrinsics.fx == 10.0


def test_frame_skip_and_max_frames(tmp_path):
    scene = sn.ScanNetScene(_write_scene(tmp_path), frame_skip=10, max_frames=1)
    assert scene.frame_ids == [0]


@pytest.mark.parametrize("ascii_ply", [False, True])
def test_ground_truth_points_from_mesh_and_aggregation(tmp_path, ascii_ply):
    scene = sn.ScanNetScene(_write_scene(tmp_path, ascii_ply=ascii_ply))
    gt = scene.ground_truth_points()
    assert gt.points.shape == (4, 3) and np.allclose(gt.points[:, 0], [0, 1, 2, 3])
    assert list(gt.labels) == ["chair", "chair", "wall", ""]
    assert list(gt.instance_ids) == [0, 0, 1, -1]


def test_read_ply_vertices_rejects_missing_vertex_element(tmp_path):
    path = tmp_path / "bad.ply"
    path.write_text("ply\nformat ascii 1.0\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n")
    with pytest.raises(ValueError):
        sn.read_ply_vertices(path)


def test_load_dataset_autodetects_scannet(tmp_path):
    assert isinstance(load_dataset(_write_scene(tmp_path)), sn.ScanNetScene)
    assert not sn.is_scannet_scene(tmp_path)
