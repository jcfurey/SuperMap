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
