"""Full-cloud geometric regions with optional, visibility-tested camera labels.

This path has no detector, learned embedding, model download, or network client.
Geometry is defined by world-frame voxels, never by a camera's field of view.
Region IDs identify connected surfaces; they are not semantic object identities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from semantic_mapping.types import CameraIntrinsics, Detection2D

UNKNOWN, DIRECT, PROPAGATED = 0, 1, 2


@dataclass
class DenseCloudConfig:
    input_mode: str = "snapshot"
    voxel_size: float = 0.05
    neighbor_radius: float = 0.15
    normal_radius: float = 0.20
    max_neighbors: int = 24
    normal_angle_deg: float = 25.0
    plane_tolerance: float = 0.04
    min_region_voxels: int = 4
    max_map_voxels: int = 500_000
    chunk_size: int = 4096
    camera_depth_tolerance: float = 0.05
    max_camera_time_delta: float = 0.20
    min_camera_score: float = 0.5
    label_propagation_radius: float = 0.0
    label_ttl_sec: float = 0.0
    allow_yoloe_labels: bool = False

    def __post_init__(self):
        if type(self.allow_yoloe_labels) is not bool:
            raise ValueError("allow_yoloe_labels must be a boolean")
        if self.input_mode not in {"snapshot", "scan"}:
            raise ValueError("input_mode must be snapshot or scan")
        for name in ("voxel_size", "neighbor_radius", "normal_radius", "plane_tolerance",
                     "camera_depth_tolerance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("max_camera_time_delta", "label_propagation_radius", "label_ttl_sec"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("max_neighbors", "min_region_voxels", "max_map_voxels", "chunk_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_neighbors < 3:
            raise ValueError("max_neighbors must be at least 3")
        if not math.isfinite(self.normal_angle_deg) or not 0 < self.normal_angle_deg < 90:
            raise ValueError("normal_angle_deg must be in (0, 90)")
        if not math.isfinite(self.min_camera_score) or not 0 <= self.min_camera_score <= 1:
            raise ValueError("min_camera_score must be in [0, 1]")


def rigid_transform(value: np.ndarray) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-5)):
        raise ValueError("pose must be a finite, rigid world-from-sensor 4x4 transform")
    return transform


@dataclass
class CameraLabels:
    stamp: float
    intrinsics: CameraIntrinsics
    T_world_from_camera: np.ndarray
    detections: list[Detection2D] = field(default_factory=list)
    depth: np.ndarray | None = None
    source: str = "manual"

    def validate(self, *, allow_yoloe=False):
        if self.source != "manual" and not (allow_yoloe and self.source == "yoloe"):
            raise ValueError("only manual camera annotations are enabled unless YOLOE is explicitly allowed")
        if not math.isfinite(self.stamp):
            raise ValueError("camera stamp must be finite")
        rigid_transform(self.T_world_from_camera)
        intr = self.intrinsics
        if (not np.isfinite([intr.fx, intr.fy, intr.cx, intr.cy]).all()
                or min(intr.fx, intr.fy) <= 0
                or not isinstance(intr.width, (int, np.integer))
                or not isinstance(intr.height, (int, np.integer))
                or min(intr.width, intr.height) <= 0):
            raise ValueError("invalid camera intrinsics")
        shape = (intr.height, intr.width)
        if self.depth is not None and np.asarray(self.depth).shape != shape:
            raise ValueError("camera depth must match the calibrated image dimensions")
        for detection in self.detections:
            if not detection.label.strip() or detection.label == "unknown":
                raise ValueError("annotations must name a nonempty semantic class")
            if not math.isfinite(detection.score) or not 0 <= detection.score <= 1:
                raise ValueError("annotation score must be in [0, 1]")
            if detection.mask is None:
                raise ValueError("camera annotations require masks; a box alone does not establish membership")
            detection.validate_mask(shape)


@dataclass
class CloudResult:
    stamp: float
    points_world: np.ndarray
    valid: np.ndarray
    region_ids: np.ndarray
    semantic_ids: np.ndarray
    confidence: np.ndarray
    label_source: np.ndarray
    camera_visible: np.ndarray
    labels: dict[int, str]
    map_points_world: np.ndarray
    map_region_ids: np.ndarray
    map_semantic_ids: np.ndarray
    map_confidence: np.ndarray
    map_label_source: np.ndarray
    stats: dict


def _cell_keys(cells):
    return np.ascontiguousarray(cells, dtype=np.int64).view(
        np.dtype([("x", "<i8"), ("y", "<i8"), ("z", "<i8")])).reshape(-1)


def _cell_match(old, new):
    """Indices of identical cells in two lexicographically sorted unique arrays."""
    if not len(old) or not len(new):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    old_keys, new_keys = _cell_keys(old), _cell_keys(new)
    index = np.searchsorted(old_keys, new_keys)
    current = np.flatnonzero(index < len(old_keys))
    current = current[old_keys[index[current]] == new_keys[current]]
    return index[current], current


def segment_surfaces(points: np.ndarray, config: DenseCloudConfig) -> np.ndarray:
    """Bounded-neighbor smooth-surface graph, computed in chunks on CPU."""
    n = len(points)
    if not n:
        return np.zeros(0, dtype=np.int32)
    tree = cKDTree(points)
    normals = np.zeros((n, 3), dtype=np.float64)
    reliable = np.zeros(n, dtype=bool)
    k = min(n, config.max_neighbors)
    for start in range(0, n, config.chunk_size):
        stop = min(start+config.chunk_size, n)
        distances, indices = tree.query(points[start:stop], k=list(range(1, k+1)),
                                        distance_upper_bound=config.normal_radius, workers=1)
        good = np.isfinite(distances)
        count = good.sum(axis=1)
        neighbors = points[np.minimum(indices, n-1)]
        mean = (neighbors*good[..., None]).sum(axis=1)/np.maximum(count, 1)[:, None]
        centered = (neighbors-mean[:, None])*good[..., None]
        covariance = np.einsum("nki,nkj->nij", centered, centered)/np.maximum(count, 1)[:, None, None]
        values, vectors = np.linalg.eigh(covariance)
        normals[start:stop] = vectors[:, :, 0]
        reliable[start:stop] = ((count >= 3) & (values[:, 1] > 1e-10)
                               & (values[:, 0] <= 0.1*np.maximum(values.sum(axis=1), 1e-12)))
    rows, cols = [], []
    cosine = math.cos(math.radians(config.normal_angle_deg))
    for start in range(0, n, config.chunk_size):
        stop = min(start+config.chunk_size, n)
        distance, neighbor = tree.query(points[start:stop], k=list(range(1, k+1)),
                                        distance_upper_bound=config.neighbor_radius, workers=1)
        row = np.broadcast_to(np.arange(start, stop)[:, None], neighbor.shape)
        valid = np.isfinite(distance) & (neighbor != row)
        a, b = row[valid], neighbor[valid]
        smooth = reliable[a] & reliable[b]
        delta = points[b]-points[a]
        aligned = np.abs(np.einsum("ni,ni->n", normals[a], normals[b])) >= cosine
        flat_a = np.abs(np.einsum("ni,ni->n", normals[a], delta)) <= config.plane_tolerance
        flat_b = np.abs(np.einsum("ni,ni->n", normals[b], delta)) <= config.plane_tolerance
        keep = ~smooth | (aligned & flat_a & flat_b)
        rows.append(a[keep]); cols.append(b[keep])
    row, col = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(row), dtype=np.uint8), (row, col)), shape=(n, n)).tocsr()
    _, component = connected_components(graph, directed=False)
    sizes = np.bincount(component)
    keep = np.flatnonzero(sizes >= config.min_region_voxels)
    mapping = np.full(len(sizes), -1, dtype=np.int32)
    mapping[keep] = np.arange(len(keep), dtype=np.int32)
    return mapping[component]


class DenseCloudPipeline:
    """Process authoritative map snapshots or accumulate registered scans.

    Missing camera coverage never deletes geometry or counts as a negative
    observation. Snapshot mode replaces the map's geometry; scan mode unions
    world voxels. A capacity error is explicit and leaves the previous state
    intact. Scan accumulation is for static mapping, without free-space carving.
    """

    def __init__(self, config: DenseCloudConfig | None = None):
        self.config = config or DenseCloudConfig()
        self.cells = np.empty((0, 3), np.int64)
        self.points = np.empty((0, 3), np.float64)
        self.regions = np.empty(0, np.int32)
        self.semantic = np.empty(0, np.int32)
        self.confidence = np.empty(0, np.float32)
        self.label_stamp = np.empty(0, np.float64)
        self.labels = {0: "unknown"}
        self._next_region = 1
        self.stamp = -math.inf
        self._input = np.empty((0, 3), np.float64)
        self._input_map = np.empty(0, np.int64)
        self._visible = np.empty(0, bool)
        self._last_camera = None
        self._annotation_sources = set()

    def _stable_regions(self, components, old_region):
        result = np.full(len(components), -1, dtype=np.int32)
        pairs_valid = (components >= 0) & (old_region >= 0)
        pairs, counts = np.unique(np.column_stack([components[pairs_valid], old_region[pairs_valid]]),
                                  axis=0, return_counts=True)
        assigned, used = {}, set()
        for index in np.argsort(-counts, kind="stable"):
            component, previous = map(int, pairs[index])
            if component not in assigned and previous not in used:
                assigned[component] = previous
                used.add(previous)
        next_region = self._next_region
        for component in np.unique(components[components >= 0]):
            if int(component) not in assigned:
                if next_region >= np.iinfo(np.int32).max:
                    raise OverflowError("region ID space exhausted; start a new map")
                assigned[int(component)] = next_region
                next_region += 1
        if assigned:
            mapping = np.full(int(components.max())+1, -1, dtype=np.int32)
            for component, identity in assigned.items():
                mapping[component] = identity
            valid = components >= 0
            result[valid] = mapping[components[valid]]
        return result, next_region

    def update(self, xyz: np.ndarray, stamp: float, T_world_from_cloud=None) -> CloudResult:
        if not math.isfinite(stamp) or stamp < self.stamp:
            raise ValueError("cloud timestamps must be finite and nondecreasing; reset after a clock jump")
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("cloud must have shape (N, 3)")
        transform = rigid_transform(np.eye(4) if T_world_from_cloud is None else T_world_from_cloud)
        points = xyz.copy()
        finite = np.isfinite(points).all(axis=1)
        points[finite] = points[finite] @ transform[:3, :3].T + transform[:3, 3]
        finite &= np.isfinite(points).all(axis=1)
        scaled = np.floor(points[finite]/self.config.voxel_size)
        if scaled.size and np.max(np.abs(scaled)) >= 2**62:
            raise ValueError("cloud coordinates exceed voxel index range")
        cells, representative, inverse = np.unique(scaled.astype(np.int64), axis=0,
                                                    return_index=True, return_inverse=True)
        current_points = points[finite][representative]
        if self.config.input_mode == "scan":
            all_cells = np.unique(np.concatenate([self.cells, cells]), axis=0)
        else:
            all_cells = cells
        if len(all_cells) > self.config.max_map_voxels:
            raise ValueError(f"map needs {len(all_cells)} voxels; max_map_voxels={self.config.max_map_voxels}; no points were silently cropped")
        old, new = _cell_match(self.cells, all_cells)
        cloud_cell, map_cell = _cell_match(cells, all_cells)
        new_points = np.empty((len(all_cells), 3), np.float64)
        new_points[new] = self.points[old]
        new_points[map_cell] = current_points[cloud_cell]
        old_regions = np.full(len(all_cells), -1, np.int32)
        old_regions[new] = self.regions[old]
        components = segment_surfaces(new_points, self.config)
        regions, next_region = self._stable_regions(components, old_regions)
        semantic = np.zeros(len(all_cells), np.int32)
        confidence = np.zeros(len(all_cells), np.float32)
        label_stamp = np.full(len(all_cells), -math.inf)
        semantic[new], confidence[new], label_stamp[new] = self.semantic[old], self.confidence[old], self.label_stamp[old]
        if self.config.label_ttl_sec > 0:
            stale = stamp-label_stamp > self.config.label_ttl_sec
            semantic[stale], confidence[stale] = 0, 0
        to_map = np.empty(len(cells), np.int64)
        to_map[cloud_cell] = map_cell
        input_map = np.full(len(points), -1, np.int64)
        input_map[finite] = to_map[inverse]
        # Commit only after validation, capacity checks, and segmentation succeed.
        self.cells, self.points, self.regions = all_cells, new_points, regions
        self.semantic, self.confidence, self.label_stamp = semantic, confidence, label_stamp
        self._next_region, self.stamp = next_region, float(stamp)
        self._input, self._input_map = points, input_map
        self._visible = np.zeros(len(points), dtype=bool)
        self._last_camera = None
        return self.result()

    def annotate(self, camera: CameraLabels) -> CloudResult:
        camera.validate(allow_yoloe=self.config.allow_yoloe_labels)
        if not math.isfinite(self.stamp):
            raise ValueError("a cloud must arrive before camera labels")
        if abs(camera.stamp-self.stamp) > self.config.max_camera_time_delta:
            raise ValueError("camera labels are stale or unsynchronized with the latest cloud")
        transform = np.asarray(camera.T_world_from_camera)
        valid = np.flatnonzero(self._input_map >= 0)
        # Inverse of a rigid world-from-camera transform, without mutating inputs.
        cam = (self._input[valid]-transform[:3, 3]) @ transform[:3, :3]
        front = cam[:, 2] > 1e-6
        valid, cam = valid[front], cam[front]
        intr = camera.intrinsics
        with np.errstate(over="ignore", invalid="ignore"):
            u = intr.fx*cam[:, 0]/cam[:, 2]+intr.cx
            v = intr.fy*cam[:, 1]/cam[:, 2]+intr.cy
        inside = (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
        valid, cam = valid[inside], cam[inside]
        u, v = np.floor(u[inside]).astype(np.int64), np.floor(v[inside]).astype(np.int64)
        pixel = v*intr.width+u
        depth_buffer = np.full(intr.width*intr.height, np.inf)
        np.minimum.at(depth_buffer, pixel, cam[:, 2])
        visible = cam[:, 2] <= depth_buffer[pixel]+self.config.camera_depth_tolerance
        if camera.depth is not None:
            measured = np.asarray(camera.depth)[v, u]
            visible &= (np.isfinite(measured) & (measured > 0)
                        & (np.abs(cam[:, 2]-measured) <= self.config.camera_depth_tolerance))
        valid, u, v = valid[visible], u[visible], v[visible]
        visible_points = np.zeros(len(self._input), dtype=bool)
        visible_points[valid] = True
        point_label = np.zeros(len(valid), np.int32)
        point_score = np.zeros(len(valid), np.float32)
        conflicted = np.zeros(len(valid), bool)
        label_lookup = {label: ident for ident, label in self.labels.items()}
        new_labels = dict(self.labels)
        for detection in camera.detections:
            if detection.score < self.config.min_camera_score:
                continue
            label = label_lookup.get(detection.label)
            if label is None:
                label = len(new_labels)
                new_labels[label] = detection.label
                label_lookup[detection.label] = label
            selected = detection.mask[v, u]
            # Overlapping classes are ambiguous regardless of detection order.
            conflicted |= selected & (point_label != 0) & (point_label != label)
            selected &= (point_label == 0) | (point_label == label)
            point_label[selected] = label
            point_score[selected] = np.maximum(point_score[selected], detection.score)
        point_label[conflicted], point_score[conflicted] = 0, 0
        selected = point_label > 0
        map_index = self._input_map[valid[selected]]
        # A voxel containing conflicting labels remains unresolved this frame.
        pairs = np.column_stack([map_index, point_label[selected]])
        unique = np.unique(pairs, axis=0)
        voxel, count = np.unique(unique[:, 0], return_counts=True)
        ambiguous = voxel[count > 1]
        accepted = ~np.isin(map_index, ambiguous)
        map_index = map_index[accepted]
        labels = point_label[selected][accepted]
        scores = point_score[selected][accepted]
        current = camera.stamp >= self.label_stamp[map_index]
        map_index, labels, scores = map_index[current], labels[current], scores[current]
        self.semantic[map_index] = labels
        self.confidence[np.unique(map_index)] = 0
        np.maximum.at(self.confidence, map_index, scores)
        self.label_stamp[map_index] = camera.stamp
        self.labels, self._visible, self._last_camera = new_labels, visible_points, float(camera.stamp)
        self._annotation_sources.add(camera.source)
        return self.result()

    def result(self) -> CloudResult:
        semantic, confidence = self.semantic.copy(), self.confidence.copy()
        source = np.where(semantic > 0, DIRECT, UNKNOWN).astype(np.uint8)
        radius = self.config.label_propagation_radius
        if radius > 0:
            # Nearest direct evidence, in the same current geometric region.
            # Propagated evidence is never stored or used as a new propagation seed.
            seeds = np.flatnonzero((semantic > 0) & (self.regions >= 0))
            target = np.flatnonzero((semantic == 0) & (self.regions >= 0))
            if len(seeds) and len(target):
                distance, nearest = cKDTree(self.points[seeds]).query(self.points[target], workers=1)
                seed = seeds[nearest]
                keep = (distance <= radius) & (self.regions[seed] == self.regions[target])
                target, seed, distance = target[keep], seed[keep], distance[keep]
                semantic[target] = semantic[seed]
                confidence[target] = confidence[seed]*(1-distance/radius)
                source[target] = PROPAGATED
        n = len(self._input)
        valid = self._input_map >= 0
        regions_out = np.full(n, -1, np.int32)
        semantics_out = np.zeros(n, np.int32)
        confidence_out = np.zeros(n, np.float32)
        source_out = np.zeros(n, np.uint8)
        index = self._input_map[valid]
        regions_out[valid], semantics_out[valid] = self.regions[index], semantic[index]
        confidence_out[valid], source_out[valid] = confidence[index], source[index]
        stats = {"input_mode": self.config.input_mode, "input_points": n,
                 "valid_input_points": int(valid.sum()), "map_voxels": len(self.points),
                 "regions": len(np.unique(self.regions[self.regions >= 0])),
                 "camera_visible_points": int(self._visible.sum()),
                 "camera_stamp": self._last_camera,
                 "direct_labeled_voxels": int((self.semantic > 0).sum()),
                 "propagated_voxels": int((source == PROPAGATED).sum()),
                 "pretrained_models": [], "annotation_sources": sorted(self._annotation_sources),
                 "geometry_is_semantic_instances": False}
        return CloudResult(self.stamp, self._input.copy(), valid.copy(), regions_out,
                           semantics_out, confidence_out, source_out, self._visible.copy(),
                           dict(self.labels), self.points.copy(), self.regions.copy(), semantic,
                           confidence, source, stats)
