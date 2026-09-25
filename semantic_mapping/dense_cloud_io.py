"""Portable annotation/result I/O and lossless PointCloud2 field extension."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from semantic_mapping.dense_cloud import CameraLabels, CloudResult
from semantic_mapping.types import CameraIntrinsics, Detection2D

ANNOTATION_FIELDS = ("region_id", "semantic_id", "semantic_confidence", "label_source", "camera_visible", "valid_geometry")


def camera_labels_from_dict(payload: dict, *, intrinsics=None, T_world_from_camera=None) -> CameraLabels:
    """Decode manual masks on a calibrated, rectified pinhole image grid.

    Each detection supplies flattened row-major ``mask_indices`` or pairs of
    ``[start, length]`` in ``mask_runs``. Runs are foreground spans, not COCO RLE.
    Callers may provide TF-derived pose and CameraInfo-derived intrinsics.
    """
    if payload.get("source") != "manual":
        raise ValueError("source must be manual; learned annotation sources are not approved")
    if payload.get("rectified") is not True:
        raise ValueError("annotations must declare rectified=true and use matching pinhole calibration")
    if intrinsics is None:
        intrinsics = CameraIntrinsics(**payload["intrinsics"])
    if T_world_from_camera is None:
        T_world_from_camera = np.asarray(payload["T_world_from_camera"], dtype=np.float64)
    camera = CameraLabels(float(payload["stamp"]), intrinsics, T_world_from_camera,
                          source=payload["source"])
    camera.validate()
    size = intrinsics.width*intrinsics.height
    if size > 100_000_000:
        raise ValueError("camera image exceeds 100 million pixels")
    detections = []
    for item in payload.get("detections", []):
        if not isinstance(item.get("label"), str):
            raise ValueError("annotation label must be a string")
        mask = np.zeros(size, dtype=bool)
        has_indices, has_runs = "mask_indices" in item, "mask_runs" in item
        if has_indices == has_runs:
            raise ValueError("each detection needs exactly one of mask_indices or mask_runs")
        if has_indices:
            indices = np.asarray(item["mask_indices"])
            if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
                raise ValueError("mask_indices must be a flat integer array")
            if indices.size and ((indices < 0).any() or (indices >= size).any()):
                raise ValueError("mask index outside the calibrated image")
            mask[indices.astype(np.int64)] = True
        else:
            for pair in item["mask_runs"]:
                if (len(pair) != 2 or any(type(v) is not int for v in pair)
                        or pair[0] < 0 or pair[1] <= 0 or pair[0]+pair[1] > size):
                    raise ValueError("mask_runs must contain valid integer [start, positive length] spans")
                mask[pair[0]:pair[0]+pair[1]] = True
        mask = mask.reshape(intrinsics.height, intrinsics.width)
        y, x = np.nonzero(mask)
        box = np.array([x.min(), y.min(), x.max()+1, y.max()+1], dtype=float) if x.size else np.zeros(4)
        detections.append(Detection2D(box, str(item["label"]), float(item.get("score", 1.0)), mask=mask))
    camera.detections = detections
    if "depth" in payload:
        camera.depth = np.asarray(payload["depth"], dtype=np.float64)
    camera.validate()
    return camera


def result_metadata(result: CloudResult) -> dict:
    """Regions are surfaces, and class distributions may be mixed within one region."""
    regions = []
    ids = result.map_region_ids
    order = np.argsort(ids, kind="stable")
    unique, start, count = np.unique(ids[order], return_index=True, return_counts=True)
    for ident, offset, length in zip(unique, start, count):
        if ident < 0:
            continue
        index = order[offset:offset+length]
        points = result.map_points_world[index]
        label, label_count = np.unique(result.map_semantic_ids[index], return_counts=True)
        regions.append({"id": int(ident), "kind": "geometric_region", "voxel_count": int(length),
                        "bbox3d": np.r_[points.min(axis=0), points.max(axis=0)].tolist(),
                        "label_voxel_counts": {result.labels[int(k)]: int(v) for k, v in zip(label, label_count)}})
    return {"schema_version": 1, "stamp": result.stamp, "labels": result.labels,
            "label_source_codes": {0: "unknown", 1: "direct_camera_voxel_evidence", 2: "propagated"},
            "stats": result.stats, "regions": regions}


def save_result(result: CloudResult, path: str | Path) -> None:
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("result path must end in .npz")
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, xyz=result.points_world, valid=result.valid,
                        region_id=result.region_ids, semantic_id=result.semantic_ids,
                        confidence=result.confidence, label_source=result.label_source,
                        camera_visible=result.camera_visible,
                        map_xyz=result.map_points_world, map_region_id=result.map_region_ids,
                        map_semantic_id=result.map_semantic_ids, map_confidence=result.map_confidence,
                        map_label_source=result.map_label_source)
    path.with_suffix(".json").write_text(json.dumps(result_metadata(result), indent=2)+"\n")


_ROS_TYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def _records(msg):
    names, formats, offsets = [], [], []
    endian = ">" if msg.is_bigendian else "<"
    for item in msg.fields:
        if item.name in names or item.datatype not in _ROS_TYPES or item.count < 1:
            raise ValueError("invalid or duplicate point cloud field")
        dtype = np.dtype(endian+_ROS_TYPES[item.datatype])
        if item.offset < 0 or item.offset+dtype.itemsize*item.count > msg.point_step:
            raise ValueError("point cloud field extends beyond point_step")
        names.append(item.name)
        formats.append(dtype if item.count == 1 else (dtype, (item.count,)))
        offsets.append(item.offset)
    if msg.row_step < msg.width*msg.point_step or len(msg.data) < msg.height*msg.row_step:
        raise ValueError("invalid point cloud row layout")
    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": msg.point_step})
    if not msg.width or not msg.height:
        return np.empty((msg.height, msg.width), dtype=dtype)
    return np.ndarray((msg.height, msg.width), dtype=dtype, buffer=bytes(msg.data),
                      strides=(msg.row_step, msg.point_step))


def full_pointcloud_xyz(msg) -> np.ndarray:
    """Retain every point in row-major order, including NaNs and organized clouds."""
    record = _records(msg)
    for name in ("x", "y", "z"):
        if name not in record.dtype.names or record.dtype[name].shape:
            raise ValueError("cloud needs scalar x, y, z fields")
        if record.dtype[name].kind != "f":
            raise ValueError("cloud xyz fields must be floating point")
    return np.column_stack([record[name].reshape(-1) for name in ("x", "y", "z")]).astype(np.float64)


def annotate_pointcloud_message(msg, result: CloudResult):
    """Append labels; preserve original point bytes, organization, fields and frame.

    Per-row padding is repacked, without changing any point's original bytes.
    Segmentation is performed in world coordinates; this output intentionally
    remains in the input cloud's frame with exactly the same point order.
    """
    from sensor_msgs.msg import PointCloud2, PointField

    record = _records(msg)
    count = msg.width*msg.height
    if len(result.region_ids) != count:
        raise ValueError("result point count differs from the original cloud")
    names = ANNOTATION_FIELDS
    if set(names) & set(record.dtype.names):
        raise ValueError("incoming cloud already has segmentation fields; use the original source topic")
    values = (result.region_ids, result.semantic_ids, result.confidence,
              result.label_source, result.camera_visible.astype(np.uint8), result.valid.astype(np.uint8))
    codes = (PointField.INT32, PointField.INT32, PointField.FLOAT32,
             PointField.UINT8, PointField.UINT8, PointField.UINT8)
    offset = msg.point_step
    fields, layouts = list(copy.deepcopy(msg.fields)), []
    endian = ">" if msg.is_bigendian else "<"
    for name, code in zip(names, codes):
        dtype = np.dtype(endian+_ROS_TYPES[code])
        fields.append(PointField(name=name, offset=offset, datatype=code, count=1))
        layouts.append((name, dtype, offset))
        offset += dtype.itemsize
    new_step = (offset+3)//4*4
    payload = np.zeros((count, new_step), dtype=np.uint8)
    if count:
        original = np.ndarray((msg.height, msg.width, msg.point_step), dtype=np.uint8,
                              buffer=bytes(msg.data), strides=(msg.row_step, msg.point_step, 1))
        payload[:, :msg.point_step] = original.reshape(count, msg.point_step)
    dtype = np.dtype({"names": names, "formats": [x[1] for x in layouts],
                      "offsets": [x[2] for x in layouts], "itemsize": new_step})
    out_record = payload.view(dtype).reshape(-1)
    for name, value in zip(names, values):
        out_record[name] = value
    return PointCloud2(header=copy.deepcopy(msg.header), height=msg.height, width=msg.width,
                       fields=fields, is_bigendian=msg.is_bigendian, point_step=new_step,
                       row_step=msg.width*new_step, data=payload.tobytes(), is_dense=bool(result.valid.all()))


def voxel_map_message(result: CloudResult, header):
    """A world-frame voxel map alongside the complete dense input-cloud output."""
    from sensor_msgs.msg import PointField
    from sensor_msgs_py import point_cloud2

    fields = [PointField(name=name, offset=i*4, datatype=PointField.FLOAT32, count=1)
              for i, name in enumerate(("x", "y", "z"))]
    fields += [PointField(name="region_id", offset=12, datatype=PointField.INT32, count=1),
               PointField(name="semantic_id", offset=16, datatype=PointField.INT32, count=1),
               PointField(name="semantic_confidence", offset=20, datatype=PointField.FLOAT32, count=1),
               PointField(name="label_source", offset=24, datatype=PointField.UINT32, count=1)]
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("region_id", "<i4"),
                      ("semantic_id", "<i4"), ("semantic_confidence", "<f4"), ("label_source", "<u4")])
    records = np.empty(len(result.map_points_world), dtype=dtype)
    for axis, name in enumerate(("x", "y", "z")):
        records[name] = result.map_points_world[:, axis]
    records["region_id"], records["semantic_id"] = result.map_region_ids, result.map_semantic_ids
    records["semantic_confidence"], records["label_source"] = result.map_confidence, result.map_label_source
    return point_cloud2.create_cloud(header, fields, records, point_step=28)
