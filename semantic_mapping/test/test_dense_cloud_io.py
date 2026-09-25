"""Annotation schema and preservation of real PointCloud2 payloads."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from semantic_mapping.dense_cloud import DenseCloudConfig, DenseCloudPipeline
from semantic_mapping.dense_cloud_io import (
    annotate_pointcloud_message, camera_labels_from_dict, full_pointcloud_xyz,
    result_metadata, save_result, voxel_map_message,
)


def payload():
    return {"source": "manual", "rectified": True, "stamp": 1.,
            "intrinsics": {"fx": 10., "fy": 10., "cx": 5., "cy": 5., "width": 10, "height": 10},
            "T_world_from_camera": np.eye(4).tolist(),
            "detections": [{"label": "chair", "score": .9, "mask_runs": [[55, 2], [65, 2]]}]}


def test_manual_mask_run_schema_and_no_model_provider():
    data = payload()
    observation = camera_labels_from_dict(data)
    assert np.flatnonzero(observation.detections[0].mask).tolist() == [55, 56, 65, 66]
    data["source"] = "unverified_model"
    with pytest.raises(ValueError, match="source must be manual"):
        camera_labels_from_dict(data)


@pytest.mark.parametrize("bad", ["negative_run", "overflow_run", "float_run", "both", "distorted", "bad_index"])
def test_invalid_annotation_schema_fails_explicitly(bad):
    data = payload()
    if bad == "negative_run":
        data["detections"][0]["mask_runs"] = [[-1, 2]]
    elif bad == "overflow_run":
        data["detections"][0]["mask_runs"] = [[99, 2]]
    elif bad == "float_run":
        data["detections"][0]["mask_runs"] = [[55.5, 2]]
    elif bad == "both":
        data["detections"][0]["mask_indices"] = [55]
    elif bad == "distorted":
        data["rectified"] = False
    else:
        del data["detections"][0]["mask_runs"]
        data["detections"][0]["mask_indices"] = [100]
    with pytest.raises(ValueError):
        camera_labels_from_dict(data)


def cloud_message(big_endian=False):
    messages = pytest.importorskip("sensor_msgs.msg")
    from std_msgs.msg import Header
    pf = messages.PointField
    endian = ">" if big_endian else "<"
    dtype = np.dtype({"names": ["x", "y", "z", "reflectivity", "ring"],
                      "formats": [endian+"f4"]*3+[endian+"u2", endian+"u2"],
                      "offsets": [0, 4, 8, 12, 14], "itemsize": 20})
    data = np.zeros((2, 2), dtype=dtype)
    data["x"] = [[0., np.nan], [3., 10.]]
    data["z"] = 2.
    data["reflectivity"] = [[11, 22], [33, 44]]
    data["ring"] = [[0, 1], [2, 3]]
    # Deliberately retain nonzero intra-point and inter-row padding.
    raw = data.view(np.uint8).reshape(2, 2, 20)
    raw[:, :, 16:] = 173
    padded = np.full((2, 44), 211, np.uint8)
    padded[:, :40] = raw.reshape(2, 40)
    fields = [pf(name=n, offset=i*4, datatype=pf.FLOAT32, count=1) for i, n in enumerate(("x", "y", "z"))]
    fields += [pf(name="reflectivity", offset=12, datatype=pf.UINT16, count=1),
               pf(name="ring", offset=14, datatype=pf.UINT16, count=1)]
    return messages.PointCloud2(header=Header(frame_id="ouster"), height=2, width=2, fields=fields,
                                is_bigendian=big_endian, point_step=20, row_step=44,
                                data=padded.tobytes(), is_dense=False)


@pytest.mark.parametrize("big_endian", [False, True])
def test_full_message_retains_fields_padding_point_order_and_invalid_geometry(big_endian):
    message = cloud_message(big_endian)
    original = bytes(message.data)
    xyz = full_pointcloud_xyz(message)
    assert xyz.shape == (4, 3) and np.isnan(xyz[1, 0])
    pipeline = DenseCloudPipeline(DenseCloudConfig(min_region_voxels=1))
    pose = np.eye(4)
    pose[0, 3] = 10.
    result = pipeline.update(xyz, 1., pose)
    output = annotate_pointcloud_message(message, result)
    assert (output.height, output.width) == (2, 2)
    assert output.header.frame_id == "ouster" and output.is_bigendian == big_endian
    assert not output.is_dense
    source_rows = np.frombuffer(original, np.uint8).reshape(2, 44)[:, :40].reshape(4, 20)
    output_rows = np.frombuffer(output.data, np.uint8).reshape(4, output.point_step)
    np.testing.assert_array_equal(output_rows[:, :20], source_rows)
    np.testing.assert_equal(full_pointcloud_xyz(output), xyz)
    assert bytes(message.data) == original
    assert [x.name for x in output.fields[:5]] == [x.name for x in message.fields]
    from semantic_mapping.dense_cloud_io import _records
    labeled = _records(output)
    np.testing.assert_array_equal(labeled["region_id"].reshape(-1), result.region_ids)
    assert labeled["valid_geometry"].reshape(-1).tolist() == [1, 0, 1, 1]


def test_voxel_map_world_frame_and_label_fields():
    pytest.importorskip("sensor_msgs.msg")
    from std_msgs.msg import Header
    from semantic_mapping.dense_cloud_io import _records
    pipeline = DenseCloudPipeline(DenseCloudConfig(min_region_voxels=1))
    result = pipeline.update(np.array([[0., 0., 2.], [1., 0., 2.]]), 1.)
    message = voxel_map_message(result, Header(frame_id="map"))
    assert message.header.frame_id == "map"
    np.testing.assert_allclose(full_pointcloud_xyz(message), result.map_points_world)
    np.testing.assert_array_equal(_records(message)["semantic_id"].reshape(-1), result.map_semantic_ids)


def test_offline_result_preserves_dense_rows_and_refuses_overwrite(tmp_path):
    pipeline = DenseCloudPipeline(DenseCloudConfig(min_region_voxels=1))
    xyz = np.array([[0., 0., 2.], [np.nan, 1., 2.], [10., 0., 2.]])
    result = pipeline.update(xyz, 1.)
    output = tmp_path/"cloud.npz"
    save_result(result, output)
    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_equal(data["xyz"], xyz)
        assert len(data["region_id"]) == 3
    assert json.loads(output.with_suffix(".json").read_text())["labels"]["0"] == "unknown"
    with pytest.raises(FileExistsError):
        save_result(result, output)


def test_offline_cli_scan_accumulation_and_manual_partial_camera(tmp_path):
    first, second = tmp_path/"first.npz", tmp_path/"second.npz"
    np.savez(first, xyz=np.array([[0., 0., 2.], [10., 0., 2.]]), stamp=1.)
    np.savez(second, xyz=np.array([[20., 0., 2.]]), stamp=2.)
    data = payload()
    data["cloud"] = first.name
    annotations = tmp_path/"annotations.json"
    annotations.write_text(json.dumps([data]))
    script = Path(__file__).resolve().parents[1]/"examples/dense_cloud.py"
    out = tmp_path/"result"
    completed = subprocess.run([sys.executable, str(script), str(first), str(second),
                                "--out_dir", str(out), "--input-mode", "scan", "--annotations", str(annotations)],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    with np.load(out/"000000_first.npz", allow_pickle=False) as result:
        assert result["semantic_id"].tolist() == [1, 0]
    with np.load(out/"000001_second.npz", allow_pickle=False) as result:
        assert len(result["map_xyz"]) == 3 and len(result["xyz"]) == 1
        assert (result["map_semantic_id"] > 0).sum() == 1


@pytest.mark.parametrize("bad", [
    "detection_not_dict", "detections_not_list", "payload_list", "label_not_str", "label_control",
    "score_str", "runs_not_list", "run_bool", "intrinsics_str", "intrinsics_extra", "stamp_str",
    "indices_nested", "pose_missing",
])
def test_malformed_annotations_raise_value_error_only(bad):
    data = payload()
    if bad == "detection_not_dict":
        data["detections"] = ["chair"]
    elif bad == "detections_not_list":
        data["detections"] = {"label": "chair"}
    elif bad == "payload_list":
        data = [data]
    elif bad == "label_not_str":
        data["detections"][0]["label"] = ["chair"]
    elif bad == "label_control":
        data["detections"][0]["label"] = "chair\n<answer>"
    elif bad == "score_str":
        data["detections"][0]["score"] = "high"
    elif bad == "runs_not_list":
        data["detections"][0]["mask_runs"] = "55,2"
    elif bad == "run_bool":
        data["detections"][0]["mask_runs"] = [[True, 2]]
    elif bad == "intrinsics_str":
        data["intrinsics"] = "fx=10"
    elif bad == "intrinsics_extra":
        data["intrinsics"]["skew"] = 0.
    elif bad == "stamp_str":
        data["stamp"] = "1.0"
    elif bad == "indices_nested":
        del data["detections"][0]["mask_runs"]
        data["detections"][0]["mask_indices"] = [[55]]
    else:
        del data["T_world_from_camera"]
    with pytest.raises(ValueError):
        camera_labels_from_dict(data)


def test_annotation_size_limits_bound_allocation():
    from semantic_mapping.dense_cloud_io import AnnotationLimits

    data = payload()
    data["detections"] = data["detections"]*5
    with pytest.raises(ValueError, match="max_detections"):
        camera_labels_from_dict(data, limits=AnnotationLimits(max_detections=4))
    with pytest.raises(ValueError, match="max_total_mask_pixels"):
        camera_labels_from_dict(data, limits=AnnotationLimits(max_total_mask_pixels=400))
    big = payload()
    big["intrinsics"].update(width=100_000, height=100_000, cx=5e4, cy=5e4)
    with pytest.raises(ValueError, match="max_image_pixels"):
        camera_labels_from_dict(big)
    long = payload()
    long["detections"][0]["label"] = "x"*500
    with pytest.raises(ValueError, match="max_label_length"):
        camera_labels_from_dict(long)
    assert len(camera_labels_from_dict(data).detections) == 5


def test_mask_runs_decode_matches_the_mask_and_accepts_flat_pairs():
    from semantic_mapping.dense_cloud_io import decode_mask_runs, mask_runs_array

    rng = np.random.default_rng(0)
    mask = rng.random((13, 17)) > .6
    runs = mask_runs_array(mask)
    np.testing.assert_array_equal(decode_mask_runs(runs, mask.size).reshape(mask.shape), mask)
    np.testing.assert_array_equal(decode_mask_runs(runs.reshape(-1), mask.size).reshape(mask.shape), mask)
    for bad in ([[0, 0]], [[-1, 2]], [[mask.size-1, 2]], [[0.5, 2]]):
        with pytest.raises(ValueError):
            decode_mask_runs(np.asarray(bad), mask.size)


def test_result_json_is_strict_and_hashing_streams(tmp_path, monkeypatch):
    from semantic_mapping.dense_cloud_io import sha256_file
    import hashlib

    empty = DenseCloudPipeline().result()  # stamp is -inf before the first cloud
    output = tmp_path/"empty.npz"
    save_result(empty, output)

    def reject(constant):
        raise AssertionError(f"non-standard JSON constant {constant}")

    metadata = json.loads(output.with_suffix(".json").read_text(), parse_constant=reject)
    assert metadata["stamp"] is None
    blob = tmp_path/"blob.bin"
    blob.write_bytes(bytes(range(256))*5000)
    expected = hashlib.sha256(blob.read_bytes()).hexdigest()
    reads = []
    original_open = Path.open

    def tracking_open(self, *args, **kwargs):
        stream = original_open(self, *args, **kwargs)
        original_read = stream.read
        stream.read = lambda size=-1: reads.append(size) or original_read(size)
        return stream

    monkeypatch.setattr(Path, "open", tracking_open)
    assert sha256_file(blob, chunk_size=4096) == expected
    assert reads and all(0 < size <= 4096 for size in reads)


def test_save_result_never_leaves_a_torn_or_partial_pair(tmp_path, monkeypatch):
    import os
    from semantic_mapping import dense_cloud_io

    result = DenseCloudPipeline(DenseCloudConfig(min_region_voxels=1)).update(np.array([[0., 0., 2.]]), 1.)
    output = tmp_path/"cloud.npz"
    calls = []
    real_replace = os.replace

    def crash_on_second(src, dst):
        calls.append(dst)
        if len(calls) == 2:
            raise OSError("simulated crash before the JSON commit")
        return real_replace(src, dst)

    monkeypatch.setattr(dense_cloud_io.os, "replace", crash_on_second)
    with pytest.raises(OSError):
        save_result(result, output)
    assert output.exists() and not output.with_suffix(".json").exists()  # incomplete save: no commit marker
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cloud.npz"]  # temporaries cleaned up

    def crash_on_first(src, dst):
        raise OSError("simulated crash while writing")

    monkeypatch.setattr(dense_cloud_io.os, "replace", crash_on_first)
    with pytest.raises(OSError):
        save_result(result, tmp_path/"other.npz")
    assert not (tmp_path/"other.npz").exists() and not (tmp_path/"other.json").exists()


def test_region_metadata_is_vectorised_and_matches_a_per_region_loop():
    rng = np.random.default_rng(1)
    pipeline = DenseCloudPipeline(DenseCloudConfig(min_region_voxels=2, voxel_size=.1))
    result = pipeline.update(rng.uniform(0, 3, (2000, 3)), 1.)
    result.map_semantic_ids[:] = rng.integers(0, 3, len(result.map_semantic_ids))
    result.labels.update({1: "a", 2: "b"})
    metadata = result_metadata(result)
    for region in metadata["regions"]:
        members = result.map_region_ids == region["id"]
        points = result.map_points_world[members]
        assert region["voxel_count"] == members.sum()
        np.testing.assert_allclose(region["bbox3d"], np.r_[points.min(axis=0), points.max(axis=0)])
        ids, counts = np.unique(result.map_semantic_ids[members], return_counts=True)
        assert region["label_voxel_counts"] == {result.labels[int(k)]: int(v) for k, v in zip(ids, counts)}
    assert {r["id"] for r in metadata["regions"]} == set(np.unique(result.map_region_ids[result.map_region_ids >= 0]))


def _info_and_header(width=17, height=13, stamp_sec=3):
    messages = pytest.importorskip("sensor_msgs.msg")
    pytest.importorskip("supermap_msgs.msg")
    from std_msgs.msg import Header
    header = Header(frame_id="camera_optical")
    header.stamp.sec, header.stamp.nanosec = stamp_sec, 250_000_000
    info = messages.CameraInfo(header=header, width=width, height=height,
                               k=[20., 0., 8., 0., 21., 6., 0., 0., 1.])
    return header, info


def test_annotations_msg_round_trip_preserves_masks_labels_and_stamp():
    from semantic_mapping.dense_cloud_io import annotations_to_msg, camera_labels_from_msg
    from semantic_mapping.types import Detection2D

    header, info = _info_and_header()
    rng = np.random.default_rng(2)
    masks = [rng.random((13, 17)) > .5, np.zeros((13, 17), bool), np.ones((13, 17), bool)]
    detections = [Detection2D(np.zeros(4), label, score, mask=mask)
                  for label, score, mask in zip(["pipe", "door", "wall"], [.9, .6, .7], masks)]
    msg = annotations_to_msg(header, info, detections, model={"name": "YOLOE", "sha": "abc"})
    assert msg.source == "yoloe" and msg.rectified and not msg.has_world_from_camera
    assert json.loads(msg.model) == {"name": "YOLOE", "sha": "abc"}
    with pytest.raises(ValueError, match="explicitly allowed"):
        camera_labels_from_msg(msg, T_world_from_camera=np.eye(4))
    with pytest.raises(ValueError, match="pose"):
        camera_labels_from_msg(msg, allow_yoloe=True)
    camera = camera_labels_from_msg(msg, T_world_from_camera=np.eye(4), allow_yoloe=True)
    assert camera.stamp == pytest.approx(3.25) and camera.source == "yoloe"
    assert (camera.intrinsics.width, camera.intrinsics.height, camera.intrinsics.fy) == (17, 13, 21.)
    for original, decoded in zip(detections, camera.detections):
        assert decoded.label == original.label and decoded.score == pytest.approx(original.score)
        np.testing.assert_array_equal(decoded.mask, original.mask)
    ys, xs = np.nonzero(masks[0])
    np.testing.assert_array_equal(camera.detections[0].bbox, [xs.min(), ys.min(), xs.max()+1, ys.max()+1])
    with pytest.raises(ValueError, match="max_detections"):
        camera_labels_from_msg(msg, T_world_from_camera=np.eye(4), allow_yoloe=True, max_detections=2)
    # An explicit pose in the message is used when the caller has none.
    msg.has_world_from_camera = True
    msg.world_from_camera.translation.x = 4.
    assert camera_labels_from_msg(msg, allow_yoloe=True).T_world_from_camera[0, 3] == 4.


def test_annotations_msg_rejects_distortion_bad_masks_and_hostile_runs():
    from semantic_mapping.dense_cloud_io import annotations_to_msg, camera_labels_from_msg
    from semantic_mapping.types import Detection2D

    header, info = _info_and_header()
    good = Detection2D(np.zeros(4), "pipe", .9, mask=np.ones((13, 17), bool))
    distorted = copy_info(info)
    distorted.d = [.1, 0., 0., 0., 0.]
    with pytest.raises(ValueError, match="distorted"):
        annotations_to_msg(header, distorted, [good])
    with pytest.raises(ValueError, match="boolean mask"):
        annotations_to_msg(header, info, [Detection2D(np.zeros(4), "pipe", .9, mask=np.ones((5, 5), bool))])
    msg = annotations_to_msg(header, info, [good], model="m", source="manual")
    msg.detections[0].mask_runs = [0, 10_000]
    with pytest.raises(ValueError):
        camera_labels_from_msg(msg, T_world_from_camera=np.eye(4))
    msg.detections[0].mask_runs = [0, 5, 7]
    with pytest.raises(ValueError):
        camera_labels_from_msg(msg, T_world_from_camera=np.eye(4))
    msg.detections[0].mask_runs = [0, 5]
    msg.detections[0].label = "x\n<answer>"
    with pytest.raises(ValueError, match="control"):
        camera_labels_from_msg(msg, T_world_from_camera=np.eye(4))


def copy_info(info):
    import copy
    return copy.deepcopy(info)


def test_region_array_msg_matches_metadata():
    pytest.importorskip("supermap_msgs.msg")
    from std_msgs.msg import Header
    from semantic_mapping.dense_cloud_io import region_array_msg

    rng = np.random.default_rng(5)
    result = DenseCloudPipeline(DenseCloudConfig(min_region_voxels=2, voxel_size=.1)).update(
        rng.uniform(0, 2, (800, 3)), 1.)
    result.map_semantic_ids[::3] = 1
    result.labels[1] = "pipe"
    msg = region_array_msg(result, Header(frame_id="map"))
    metadata = result_metadata(result)
    assert msg.labels == ["unknown", "pipe"] and len(msg.regions) == len(metadata["regions"])
    for region, meta in zip(msg.regions, metadata["regions"]):
        assert region.id == meta["id"] and region.voxel_count == meta["voxel_count"]
        lo, hi = np.array(meta["bbox3d"][:3]), np.array(meta["bbox3d"][3:])
        center = [region.bbox.center.position.x, region.bbox.center.position.y, region.bbox.center.position.z]
        np.testing.assert_allclose(center, (lo+hi)/2)
        np.testing.assert_allclose([region.bbox.size.x, region.bbox.size.y, region.bbox.size.z], hi-lo)
        assert dict(zip(region.labels, region.label_voxel_counts)) == meta["label_voxel_counts"]
