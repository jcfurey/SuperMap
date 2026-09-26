"""Persist and restore the object map M_t across sessions.

A living spatial memory has to outlive the process that built it: a robot
that mapped a building yesterday should start today knowing what it saw,
which instance IDs it assigned, and how each object's state evolved. This
module writes the complete map state -- per-instance points with their
geometric log-odds and membership evidence, label beliefs, lifecycle status,
2D tracklet, timestamps, hit counts, and the temporal-edge trajectory --
plus the ID counter, so a restored map continues exactly where it stopped
and never reuses an ID.

Layout::

    <map_dir>/map.json                  header, ID counter, and every instance's scalar fields
    <map_dir>/map_arrays-<save>.npz     points_<id>, log_odds_<id>, membership_<id> per instance
                                        (+ support_<id> when bbox support is tracked)

Saves are atomic. The arrays go to a file unique to this save, written via
a temporary file, fsync'ed, and renamed into place; ``map.json`` -- which
names that arrays file -- is replaced the same way last, and is the commit
marker. A crash at any point leaves the previous ``map.json`` and the
arrays file it names intact, so the directory always loads as either the
old or the new map. Arrays files no committed header refers to are removed
after the commit. Maps written before this scheme (a fixed
``map_arrays.npz``, no ``arrays_file`` in the header) still load.

Points are stored as float64: float32 has a 24-bit mantissa, which leaves
centimetre-level steps at world coordinates around 1e5 m (e.g. UTM
eastings). Older maps with float32 points load unchanged.

Resuming (the default when loading) treats the restored objects the way the
tracker treats anything that left the field of view: active instances come
back as *occluded* with a reset 2D tracklet, since a tracklet is
camera-relative and meaningless after a restart, while the 3D state, label,
ID, and history carry over. Re-observation then goes through the 3D
re-activation stage (Sec. IV-B), and the geometric-consistency update
retires objects that are gone.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

from semantic_mapping.object_map import ObjectMap
from semantic_mapping.tracking import current_bbox, init_track
from semantic_mapping.types import ObjectInstance, ObjectStatus, TrackKalmanState

MAP_FORMAT_VERSION = 1
MAP_JSON = "map.json"
MAP_ARRAYS = "map_arrays.npz"
"""Arrays file of maps written before atomic saves; see ``arrays_file`` in the header."""
_ARRAYS_PREFIX = "map_arrays"


def _floats(values) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=np.float64).ravel()]


def instance_to_record(obj: ObjectInstance) -> dict:
    """Everything about an instance except its per-point arrays, as JSON-safe values."""
    return {
        "instance_id": int(obj.instance_id),
        "label_belief": {str(k): float(v) for k, v in obj.label_belief.items()},
        "bbox3d": _floats(obj.bbox3d),
        "status": obj.status.value,
        "track": {"state": _floats(obj.track.state), "covariance": _floats(obj.track.covariance)},
        "first_seen_stamp": float(obj.first_seen_stamp),
        "latest_stamp": float(obj.latest_stamp),
        "frames_since_seen": int(obj.frames_since_seen),
        "missed_detection_frames": int(obj.missed_detection_frames),
        "geometry_stamp": obj.geometry_stamp,
        "hits": int(obj.hits),
        "points_contradicted": int(obj.points_contradicted),
        "existence_log_odds": float(obj.existence_log_odds),
        "trajectory": [[float(stamp), _floats(center), str(status)] for stamp, center, status in obj.trajectory],
        "embedding": _floats(obj.embedding) if obj.embedding is not None else None,
        "embedding_count": int(obj.embedding_count),
        "match_stamps": _floats(obj.match_stamps),
    }


def instance_from_record(
    record: dict, points: np.ndarray, log_odds: np.ndarray, membership: np.ndarray,
) -> ObjectInstance:
    n = points.shape[0]
    if log_odds.shape[0] != n or membership.shape[0] != n:
        raise ValueError(f"instance {record['instance_id']}: per-point arrays disagree on length")
    track = record["track"]
    return ObjectInstance(
        instance_id=int(record["instance_id"]),
        label_belief={str(k): float(v) for k, v in record["label_belief"].items()},
        points_world=np.asarray(points, dtype=np.float64).reshape(-1, 3),
        point_log_odds=np.asarray(log_odds, dtype=np.float64),
        bbox3d=np.asarray(record["bbox3d"], dtype=np.float64),
        status=ObjectStatus(record["status"]),
        track=TrackKalmanState(
            state=np.asarray(track["state"], dtype=np.float64),
            covariance=np.asarray(track["covariance"], dtype=np.float64).reshape(6, 6),
        ),
        first_seen_stamp=float(record["first_seen_stamp"]),
        latest_stamp=float(record["latest_stamp"]),
        frames_since_seen=int(record["frames_since_seen"]),
        missed_detection_frames=int(record.get("missed_detection_frames", 0)),
        geometry_stamp=record.get("geometry_stamp", float(record['latest_stamp']) if n else None),
        hits=int(record["hits"]),
        points_contradicted=int(record.get("points_contradicted", 0)),
        existence_log_odds=float(record.get("existence_log_odds", 0.0)),
        point_membership=np.asarray(membership, dtype=np.float64),
        trajectory=[(float(s), np.asarray(c, dtype=np.float64), str(st)) for s, c, st in record.get("trajectory", [])],
        embedding=(np.asarray(record["embedding"], dtype=np.float32) if record.get("embedding") is not None else None),
        embedding_count=int(record.get("embedding_count", 0)),
        match_stamps=[float(s) for s in record.get("match_stamps", [])],
    )


def prepare_for_resume(obj: ObjectInstance) -> None:
    """Make a restored instance safe to track from a fresh, unrelated viewpoint."""
    obj.track = init_track(current_bbox(obj.track))
    if obj.status == ObjectStatus.ACTIVE:
        obj.status = ObjectStatus.OCCLUDED


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:  # e.g. platforms that cannot open directories
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write(path: Path, write) -> None:
    """Write ``path`` via a same-directory temporary file, fsync, and os.replace."""
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_map(object_map: ObjectMap, path: str | Path, metadata: dict | None = None) -> Path:
    """Write the map to ``path`` (a directory, created if needed). Returns the directory.

    Atomic: see the module docstring.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    records = []
    for obj in sorted(object_map.objects.values(), key=lambda o: o.instance_id):
        records.append(instance_to_record(obj))
        arrays[f"points_{obj.instance_id}"] = np.asarray(obj.points_world, dtype=np.float64).reshape(-1, 3)
        arrays[f"log_odds_{obj.instance_id}"] = np.asarray(obj.point_log_odds, dtype=np.float32)
        arrays[f"membership_{obj.instance_id}"] = np.asarray(obj.point_membership, dtype=np.float32)
        if len(obj.point_support) and obj.point_support.shape[0] == obj.points_world.shape[0]:
            arrays[f"support_{obj.instance_id}"] = np.asarray(obj.point_support, dtype=np.float32)
    arrays_file = f"{_ARRAYS_PREFIX}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}.npz"
    header = {
        "format_version": MAP_FORMAT_VERSION,
        "saved_at": time.time(),
        "next_instance_id": int(object_map._next_id),
        "voxel_size": float(object_map.voxel_size),
        "num_instances": len(records),
        "arrays_file": arrays_file,
        "metadata": dict(metadata or {}),
        "instances": records,
    }
    _atomic_write(path / arrays_file, lambda handle: np.savez_compressed(handle, **arrays))
    _fsync_dir(path)
    _atomic_write(path / MAP_JSON, lambda handle: handle.write(json.dumps(header, indent=1).encode()))
    _fsync_dir(path)
    # Committed: arrays files (and temporaries) of earlier or interrupted saves are garbage now.
    for stale in path.iterdir():
        name = stale.name
        if name == arrays_file:
            continue
        if (name.startswith(_ARRAYS_PREFIX) and name.endswith(".npz")) or \
                (name.startswith(f".{_ARRAYS_PREFIX}") or name.startswith(f".{MAP_JSON}.")) and name.endswith(".tmp"):
            try:
                stale.unlink()
            except OSError:
                pass
    return path


def load_map(path: str | Path, object_map: ObjectMap, resume: bool = True) -> dict:
    """Replace ``object_map``'s contents with the map saved at ``path``.

    Returns the saved header (format version, timestamps, metadata, ...).
    With ``resume`` (default) instances are prepared for tracking from a new
    viewpoint, see :func:`prepare_for_resume`; ``resume=False`` restores the
    exact saved state, e.g. for round-trip checks.
    """
    path = Path(path)
    header = json.loads((path / MAP_JSON).read_text())
    version = int(header.get("format_version", -1))
    if version != MAP_FORMAT_VERSION:
        raise ValueError(f"{path}: map format version {version} is not supported (expected {MAP_FORMAT_VERSION})")

    objects: dict[int, ObjectInstance] = {}
    arrays_file = Path(str(header.get("arrays_file") or MAP_ARRAYS)).name  # never outside the map directory
    with np.load(path / arrays_file) as arrays:
        for record in header["instances"]:
            instance_id = int(record["instance_id"])
            obj = instance_from_record(
                record,
                arrays[f"points_{instance_id}"],
                arrays[f"log_odds_{instance_id}"],
                arrays[f"membership_{instance_id}"],
            )
            if f"support_{instance_id}" in arrays:
                obj.point_support = np.asarray(arrays[f"support_{instance_id}"], dtype=np.float64)
            if resume:
                prepare_for_resume(obj)
            objects[instance_id] = obj

    object_map.objects = objects
    object_map._next_id = max(int(header.get("next_instance_id", 1)), *(i + 1 for i in objects), 1)
    return header
