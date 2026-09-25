"""Portable annotation/result I/O and lossless PointCloud2 field extension.

Camera annotations travel either as typed ``supermap_msgs/CameraAnnotations``
(the live path, see :func:`annotations_to_msg` / :func:`camera_labels_from_msg`)
or as the JSON schema used by offline tools and manual labelling
(:func:`annotations_payload` / :func:`camera_labels_from_dict`). Both decoders
treat their input as untrusted: types are checked, and detection count, image
size, total mask memory and label length are bounded.
"""
from __future__ import annotations

import array
import copy
import hashlib
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from semantic_mapping.dense_cloud import CameraLabels, CloudResult, rigid_transform
from semantic_mapping.types import CameraIntrinsics, Detection2D
from semantic_mapping.ros_msgs import stamp_to_seconds

ANNOTATION_FIELDS = ("region_id", "semantic_id", "semantic_confidence", "label_source", "camera_visible", "valid_geometry")
LABEL_SOURCE_CODES = {0: "unknown", 1: "direct_camera_voxel_evidence", 2: "propagated"}


@dataclass(frozen=True)
class AnnotationLimits:
    """Bounds applied to every decoded annotation (review items C4/C28)."""
    max_detections: int = 64
    max_image_pixels: int = 16_777_216  # e.g. 4096 x 4096
    max_total_mask_pixels: int = 1 << 28  # decoded bool masks, summed over detections (256 MiB)
    max_label_length: int = 128
    max_runs_per_detection: int = 1 << 20

    def check_image(self, intrinsics) -> int:
        size = int(intrinsics.width)*int(intrinsics.height)
        if size > self.max_image_pixels:
            raise ValueError(f"camera image has {size} pixels; max_image_pixels={self.max_image_pixels}")
        return size

    def check_detection_count(self, count, size):
        if count > self.max_detections:
            raise ValueError(f"{count} detections exceed max_detections={self.max_detections}")
        if count*size > self.max_total_mask_pixels:
            raise ValueError(f"{count} masks of {size} pixels exceed max_total_mask_pixels={self.max_total_mask_pixels}")

    def check_label(self, label):
        if not isinstance(label, str) or not label.strip():
            raise ValueError("annotation label must be a nonempty string")
        if len(label) > self.max_label_length:
            raise ValueError(f"annotation label longer than max_label_length={self.max_label_length}")
        if any(ord(c) < 32 for c in label):
            raise ValueError("annotation label contains control characters")
        return label


DEFAULT_LIMITS = AnnotationLimits()


def mask_runs_array(mask) -> np.ndarray:
    """Foreground spans ``(start, length)`` of a mask in row-major order, shape (k, 2)."""
    flat = np.asarray(mask, dtype=bool).reshape(-1)
    edges = np.flatnonzero(np.diff(np.r_[False, flat, False].astype(np.int8)))
    return np.column_stack([edges[::2], edges[1::2]-edges[::2]]).astype(np.int64)


def mask_runs(mask):
    """Encode foreground spans in row-major order, including empty/full masks."""
    return mask_runs_array(mask).tolist()


def decode_mask_runs(runs, size: int, limits: AnnotationLimits = DEFAULT_LIMITS) -> np.ndarray:
    """Flat bool mask from ``(k, 2)`` or flattened ``[start, length, ...]`` spans.

    Vectorised; rejects floats, booleans, negative/zero lengths and spans that
    leave the image. Overlapping spans are tolerated (they union).
    """
    runs = np.asarray(runs)
    if runs.size == 0:
        return np.zeros(size, dtype=bool)
    if runs.dtype.kind not in "iu" or runs.dtype == np.bool_:
        raise ValueError("mask_runs must contain integer [start, positive length] spans")
    runs = runs.reshape(-1, 2) if runs.ndim == 1 and runs.size % 2 == 0 else runs
    if runs.ndim != 2 or runs.shape[1] != 2:
        raise ValueError("mask_runs must be [start, length] pairs")
    if len(runs) > limits.max_runs_per_detection:
        raise ValueError(f"mask has more than max_runs_per_detection={limits.max_runs_per_detection} spans")
    runs = runs.astype(np.int64)
    start, length = runs[:, 0], runs[:, 1]
    if (start < 0).any() or (length <= 0).any() or (start+length > size).any():
        raise ValueError("mask_runs must contain valid integer [start, positive length] spans")
    delta = np.zeros(size+1, dtype=np.int32)
    np.add.at(delta, start, 1)
    np.add.at(delta, start+length, -1)
    return np.cumsum(delta[:-1]) > 0


def _mask_box(mask):
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any():
        return np.zeros(4)
    y = np.flatnonzero(rows)
    x = np.flatnonzero(cols)
    return np.array([x[0], y[0], x[-1]+1, y[-1]+1], dtype=float)


def _model_text(model) -> str:
    if model is None:
        return ""
    if isinstance(model, str):
        return model
    return json.dumps(json_safe(model), sort_keys=True, allow_nan=False)


def annotations_payload(header, intrinsics, detections, model):
    items = []
    for detection in detections:
        if detection.mask is None:
            raise ValueError("YOLOE must return segmentation masks; boxes alone are not labels")
        detection.validate_mask((intrinsics.height, intrinsics.width))
        items.append({"label": detection.label, "score": float(detection.score),
                      "mask_runs": mask_runs(detection.mask)})
    return {"source": "yoloe", "rectified": True, "stamp": stamp_to_seconds(header.stamp),
            "frame_id": header.frame_id, "intrinsics": asdict(intrinsics),
            "detections": items, "model": model}


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    """Streamed SHA-256 of a file (never reads the whole file into memory)."""
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
        return digest.hexdigest()


def json_safe(value):
    """Recursively convert numpy scalars/arrays and map non-finite floats to ``None``.

    Standard JSON has no ``Infinity``/``NaN``; ``json.dumps`` would otherwise
    emit them (e.g. the pipeline's initial ``stamp=-inf``) and strict parsers
    reject the file.
    """
    if isinstance(value, dict):
        return {str(k) if not isinstance(k, str) else k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dumps_json(value, **kwargs) -> str:
    return json.dumps(json_safe(value), allow_nan=False, **kwargs)


def check_rectified_camera_info(info) -> None:
    """Masks are only meaningful on a rectified pinhole grid: D == 0 and R == I (or unset)."""
    d = np.asarray(info.d, dtype=np.float64)
    if not np.isfinite(d).all() or np.any(np.abs(d) > 1e-12):
        raise ValueError("distorted CameraInfo cannot describe rectified masks")
    rotation = np.asarray(info.r, dtype=np.float64).reshape(3, 3)
    if rotation.any() and not np.allclose(rotation, np.eye(3)):
        raise ValueError("nonidentity camera rectification requires explicit rectified optical calibration")


def camera_labels_from_dict(payload: dict, *, intrinsics=None, T_world_from_camera=None,
                            allow_yoloe=False, limits: AnnotationLimits = DEFAULT_LIMITS) -> CameraLabels:
    """Decode allowed masks on a calibrated, rectified pinhole image grid.

    Each detection supplies flattened row-major ``mask_indices`` or pairs of
    ``[start, length]`` in ``mask_runs``. Runs are foreground spans, not COCO RLE.
    Callers may provide TF-derived pose and CameraInfo-derived intrinsics.
    Malformed input of any shape raises ``ValueError``.
    """
    if not isinstance(payload, dict):
        raise ValueError("annotation payload must be a JSON object")
    source = payload.get("source")
    if source != "manual" and not (allow_yoloe and source == "yoloe"):
        raise ValueError("source must be manual unless YOLOE is explicitly allowed")
    if payload.get("rectified") is not True:
        raise ValueError("annotations must declare rectified=true and use matching pinhole calibration")
    if intrinsics is None:
        raw = payload.get("intrinsics")
        if not isinstance(raw, dict) or set(raw) != {"fx", "fy", "cx", "cy", "width", "height"}:
            raise ValueError("intrinsics must be an object with fx, fy, cx, cy, width, height")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in raw.values()):
            raise ValueError("intrinsics values must be numbers")
        if not all(isinstance(raw[k], int) for k in ("width", "height")):
            raise ValueError("intrinsics width and height must be integers")
        intrinsics = CameraIntrinsics(**raw)
    if T_world_from_camera is None:
        try:
            T_world_from_camera = np.asarray(payload["T_world_from_camera"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("annotation needs T_world_from_camera (or a TF pose from the caller)") from exc
    stamp = payload.get("stamp")
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
        raise ValueError("annotation stamp must be a number")
    camera = CameraLabels(float(stamp), intrinsics, T_world_from_camera, source=source)
    camera.validate(allow_yoloe=allow_yoloe)
    size = limits.check_image(intrinsics)
    items = payload.get("detections", [])
    if not isinstance(items, list):
        raise ValueError("detections must be a list")
    limits.check_detection_count(len(items), size)
    detections = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each detection must be an object")
        label = limits.check_label(item.get("label"))
        score = item.get("score", 1.0)
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("annotation score must be a number")
        has_indices, has_runs = "mask_indices" in item, "mask_runs" in item
        if has_indices == has_runs:
            raise ValueError("each detection needs exactly one of mask_indices or mask_runs")
        if has_indices:
            if not isinstance(item["mask_indices"], list):
                raise ValueError("mask_indices must be a flat integer array")
            indices = np.asarray(item["mask_indices"])
            if indices.ndim != 1 or (indices.size and (indices.dtype.kind not in "iu" or indices.dtype == np.bool_)):
                raise ValueError("mask_indices must be a flat integer array")
            if indices.size > size:
                raise ValueError("more mask indices than image pixels")
            if indices.size and ((indices < 0).any() or (indices >= size).any()):
                raise ValueError("mask index outside the calibrated image")
            mask = np.zeros(size, dtype=bool)
            mask[indices.astype(np.int64)] = True
        else:
            runs = item["mask_runs"]
            if not isinstance(runs, list) or any(not isinstance(pair, list) or len(pair) != 2
                                                 or any(type(v) is not int for v in pair) for pair in runs):
                raise ValueError("mask_runs must contain valid integer [start, positive length] spans")
            mask = decode_mask_runs(np.asarray(runs, dtype=np.int64).reshape(-1, 2), size, limits)
        mask = mask.reshape(intrinsics.height, intrinsics.width)
        detections.append(Detection2D(_mask_box(mask), label, float(score), mask=mask))
    camera.detections = detections
    if "depth" in payload:
        try:
            camera.depth = np.asarray(payload["depth"], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("depth must be a numeric image") from exc
    camera.validate(allow_yoloe=allow_yoloe)
    return camera


# ------------------------------------------------------------- typed messages
def annotations_to_msg(header, camera_info, detections, model: str = "", source: str = "yoloe"):
    """Build ``supermap_msgs/CameraAnnotations`` for masks on a rectified image.

    ``header`` is the image header (capture stamp, optical frame) and
    ``camera_info`` its calibration; masks must match the ROI/binning-adjusted
    image size. ``model`` may be a string or a metadata dict (stored as sorted
    JSON). Mask runs are flattened ``[start0, length0, start1, length1, ...]``.
    """
    from supermap_msgs.msg import CameraAnnotations, MaskDetection

    from semantic_mapping.ros_msgs import camera_info_to_intrinsics

    if source not in ("yoloe", "manual"):
        raise ValueError("source must be 'yoloe' or 'manual'")
    check_rectified_camera_info(camera_info)
    intrinsics = camera_info_to_intrinsics(camera_info)
    shape = (intrinsics.height, intrinsics.width)
    msg = CameraAnnotations()
    msg.header = copy.deepcopy(header)
    msg.source = source
    msg.rectified = True
    msg.camera_info = copy.deepcopy(camera_info)
    msg.has_world_from_camera = False
    msg.model = _model_text(model)
    items = []
    for detection in detections:
        if detection.mask is None:
            raise ValueError("camera annotations need segmentation masks; boxes alone are not labels")
        detection.validate_mask(shape)
        runs = array.array("I")
        runs.frombytes(np.ascontiguousarray(mask_runs_array(detection.mask).reshape(-1), dtype=np.uint32).tobytes())
        item = MaskDetection(label=str(detection.label), score=float(detection.score))
        item.mask_runs = runs
        items.append(item)
    msg.detections = items
    return msg


def _transform_msg_to_se3(transform):
    from semantic_mapping.geometry_utils import se3_from_translation_quaternion

    t, q = transform.translation, transform.rotation
    return se3_from_translation_quaternion(np.array([t.x, t.y, t.z]), np.array([q.x, q.y, q.z, q.w]))


def camera_labels_from_msg(msg, *, intrinsics=None, T_world_from_camera=None, allow_yoloe=False,
                           max_detections: int = DEFAULT_LIMITS.max_detections,
                           limits: AnnotationLimits | None = None) -> CameraLabels:
    """Decode ``supermap_msgs/CameraAnnotations`` into :class:`CameraLabels`.

    The pose is ``T_world_from_camera`` when given (e.g. from TF at the image
    stamp), else the message's explicit ``world_from_camera``; with neither the
    call fails. Intrinsics come from the embedded CameraInfo unless given.
    """
    limits = limits or AnnotationLimits(max_detections=max_detections)
    if msg.source != "manual" and not (allow_yoloe and msg.source == "yoloe"):
        raise ValueError("source must be manual unless YOLOE is explicitly allowed")
    if not msg.rectified:
        raise ValueError("annotations must declare rectified=true and use matching pinhole calibration")
    if intrinsics is None:
        from semantic_mapping.ros_msgs import camera_info_to_intrinsics

        check_rectified_camera_info(msg.camera_info)
        intrinsics = camera_info_to_intrinsics(msg.camera_info)
    if T_world_from_camera is None:
        if not msg.has_world_from_camera:
            raise ValueError("annotation has no explicit pose; supply T_world_from_camera from TF")
        T_world_from_camera = _transform_msg_to_se3(msg.world_from_camera)
    rigid_transform(T_world_from_camera)
    stamp = stamp_to_seconds(msg.header.stamp)
    camera = CameraLabels(stamp, intrinsics, np.asarray(T_world_from_camera, dtype=np.float64), source=msg.source)
    camera.validate(allow_yoloe=allow_yoloe)
    size = limits.check_image(intrinsics)
    limits.check_detection_count(len(msg.detections), size)
    detections = []
    for item in msg.detections:
        label = limits.check_label(item.label)
        runs = np.frombuffer(item.mask_runs, dtype=np.uint32) if isinstance(item.mask_runs, array.array) \
            else np.asarray(item.mask_runs, dtype=np.int64)
        if runs.size % 2:
            raise ValueError("mask_runs must hold [start, length] pairs")
        mask = decode_mask_runs(runs.astype(np.int64), size, limits).reshape(intrinsics.height, intrinsics.width)
        detections.append(Detection2D(_mask_box(mask), label, float(item.score), mask=mask))
    camera.detections = detections
    camera.validate(allow_yoloe=allow_yoloe)
    return camera


# ------------------------------------------------------------- region metadata
def region_statistics(result: CloudResult) -> dict:
    """Vectorised per-region voxel counts, bounds and label histograms."""
    ids = result.map_region_ids
    valid = np.flatnonzero(ids >= 0)
    if not len(valid):
        return {"ids": np.zeros(0, np.int64), "counts": np.zeros(0, np.int64),
                "min": np.zeros((0, 3)), "max": np.zeros((0, 3)), "label_pairs": np.zeros((0, 2), np.int64),
                "label_counts": np.zeros(0, np.int64), "label_offsets": np.zeros(1, np.int64)}
    order = valid[np.argsort(ids[valid], kind="stable")]
    sorted_ids = ids[order]
    unique, start, count = np.unique(sorted_ids, return_index=True, return_counts=True)
    points = result.map_points_world[order]
    low = np.minimum.reduceat(points, start, axis=0)
    high = np.maximum.reduceat(points, start, axis=0)
    region_index = np.repeat(np.arange(len(unique)), count)
    pairs, pair_counts = np.unique(np.column_stack([region_index, result.map_semantic_ids[order]]),
                                   axis=0, return_counts=True)
    offsets = np.searchsorted(pairs[:, 0], np.arange(len(unique)+1))
    return {"ids": unique, "counts": count, "min": low, "max": high, "label_pairs": pairs,
            "label_counts": pair_counts, "label_offsets": offsets}


def result_metadata(result: CloudResult) -> dict:
    """Regions are surfaces, and class distributions may be mixed within one region."""
    stats = region_statistics(result)
    regions = []
    labels = result.labels
    bounds = np.hstack([stats["min"], stats["max"]]).tolist()
    for i, (ident, length) in enumerate(zip(stats["ids"].tolist(), stats["counts"].tolist())):
        lo, hi = stats["label_offsets"][i], stats["label_offsets"][i+1]
        regions.append({"id": ident, "kind": "geometric_region", "voxel_count": length,
                        "bbox3d": bounds[i],
                        "label_voxel_counts": {labels[int(k)]: int(v) for k, v in
                                               zip(stats["label_pairs"][lo:hi, 1], stats["label_counts"][lo:hi])}})
    return json_safe({"schema_version": 1, "stamp": result.stamp, "labels": result.labels,
                      "label_source_codes": LABEL_SOURCE_CODES,
                      "stats": result.stats, "regions": regions})


def region_array_msg(result: CloudResult, header):
    """``supermap_msgs/RegionArray`` for the map (header frame = world frame)."""
    from supermap_msgs.msg import Region, RegionArray
    from vision_msgs.msg import BoundingBox3D

    stats = region_statistics(result)
    vocabulary = [result.labels[k] for k in sorted(result.labels)]
    centers = ((stats["min"]+stats["max"])/2).tolist()
    sizes = (stats["max"]-stats["min"]).tolist()
    regions = []
    for i, (ident, length) in enumerate(zip(stats["ids"].tolist(), stats["counts"].tolist())):
        lo, hi = stats["label_offsets"][i], stats["label_offsets"][i+1]
        box = BoundingBox3D()
        box.center.position.x, box.center.position.y, box.center.position.z = centers[i]
        box.center.orientation.w = 1.0
        box.size.x, box.size.y, box.size.z = sizes[i]
        region = Region(id=int(ident), voxel_count=int(length), bbox=box,
                        labels=[result.labels[int(k)] for k in stats["label_pairs"][lo:hi, 1]])
        region.label_voxel_counts = array.array("I", stats["label_counts"][lo:hi].astype(np.uint32).tolist())
        regions.append(region)
    return RegionArray(header=header, labels=vocabulary, regions=regions)


# ------------------------------------------------------------------ saving
def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def save_result(result: CloudResult, path: str | Path) -> None:
    """Write ``<name>.npz`` plus ``<name>.json`` without leaving a torn pair.

    Both files are written to temporary names in the target directory,
    flushed, and then renamed into place: the NPZ first, the JSON last. The
    JSON is the commit marker, so a crash can leave at most an NPZ without its
    JSON (an incomplete save), never a JSON describing other arrays, and never
    a partially written file under the final name.
    """
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("result path must end in .npz")
    meta_path = path.with_suffix(".json")
    if path.exists() or meta_path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dumps_json(result_metadata(result), indent=2)+"\n"
    temps = []
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.stem}.", suffix=".npz.tmp",
                                         delete=False) as stream:
            temps.append(Path(stream.name))
            np.savez_compressed(stream, xyz=result.points_world, valid=result.valid,
                                region_id=result.region_ids, semantic_id=result.semantic_ids,
                                confidence=result.confidence, label_source=result.label_source,
                                camera_visible=result.camera_visible,
                                map_xyz=result.map_points_world, map_region_id=result.map_region_ids,
                                map_semantic_id=result.map_semantic_ids, map_confidence=result.map_confidence,
                                map_label_source=result.map_label_source)
            stream.flush()
            os.fsync(stream.fileno())
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.stem}.", suffix=".json.tmp",
                                         delete=False, encoding="utf-8") as stream:
            temps.append(Path(stream.name))
            stream.write(metadata)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() or meta_path.exists():
            raise FileExistsError(path)
        os.replace(temps[0], path)
        os.replace(temps[1], meta_path)
        _fsync_dir(path.parent)
    finally:
        for temp in temps:
            temp.unlink(missing_ok=True)


_ROS_TYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def _buffer(data):
    """Zero-copy view of message bytes where the container supports it."""
    if isinstance(data, (bytes, bytearray, memoryview, array.array, np.ndarray)):
        return data
    return bytes(data)


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
    return np.ndarray((msg.height, msg.width), dtype=dtype, buffer=_buffer(msg.data),
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
                              buffer=_buffer(msg.data), strides=(msg.row_step, msg.point_step, 1))
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
