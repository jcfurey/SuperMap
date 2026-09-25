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

from semantic_mapping import native
from semantic_mapping.geometry_utils import GROUND_SURFACE_LABELS, fit_ground_plane, pack_voxel_keys, splat_depth_buffer
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
    # Threads for neighbour searches and the surface graph, the bulk of
    # segmentation time; -1 uses every core. Results do not depend on it.
    # Lower it to leave cores for other processes on the robot.
    workers: int = -1
    # Build the surface graph with the compiled supermap_kernels package when
    # it is installed (semantic_mapping.native); NumPy/SciPy otherwise. The
    # two agree except where equally distant neighbours tie for the k-th place.
    native_kernels: bool = True
    camera_depth_tolerance: float = 0.05
    max_camera_time_delta: float = 0.20
    min_camera_score: float = 0.5
    label_propagation_radius: float = 0.0
    label_ttl_sec: float = 0.0
    allow_yoloe_labels: bool = False
    # Occlusion through sparse LiDAR (review C5). Each projected point covers a
    # disc of this physical radius (so f*r/z pixels, capped) in the z-buffer;
    # a point is hidden when another point's footprint is more than
    # camera_splat_occlusion_gap metres in front of it. 0 radius disables it.
    camera_splat_radius: float = 0.05
    camera_max_splat_pixels: int = 8
    camera_splat_occlusion_gap: float = 0.30
    # Per-mask foreground layer: in-mask points are sorted by depth and cut at
    # the first gap larger than max(mask_depth_gap, mask_depth_gap_ratio*depth);
    # only the nearest layer with >= mask_min_layer_points points takes the
    # label, so background seen "through" a mask stays unknown. 0 gap disables.
    mask_depth_gap: float = 1.0
    mask_depth_gap_ratio: float = 0.25
    mask_min_layer_points: int = 3
    # A mask bleeds onto the ground at an object's base, and a spinning
    # LiDAR's ground rings there are nearer than the object: without this the
    # label lands on the ring in front of the object and slides along with the
    # sensor. Visible points less than ground_clearance above the local ground
    # (world z up, fitted around the mask) do not take a mask's label when at
    # least mask_min_layer_points of its points stand above it. Masks of
    # ground_surface_labels, masks with no fitted ground below the camera and
    # masks lying flat on the ground keep their ground points.
    ground_exclusion: bool = True
    ground_clearance: float = 0.15
    ground_max_slope: float = 0.25
    ground_context_px: int = 20
    ground_surface_labels: list = field(default_factory=lambda: list(GROUND_SURFACE_LABELS))
    # Bound on distinct semantic labels in one map (untrusted annotations, C28).
    max_labels: int = 1024
    # Re-segment only voxels near changed geometry (P2); falls back to a full
    # pass when most of the map changed.
    incremental_segmentation: bool = True

    def __post_init__(self):
        for name in ("allow_yoloe_labels", "incremental_segmentation", "ground_exclusion", "native_kernels"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.input_mode not in {"snapshot", "scan"}:
            raise ValueError("input_mode must be snapshot or scan")
        for name in ("voxel_size", "neighbor_radius", "normal_radius", "plane_tolerance",
                     "camera_depth_tolerance", "ground_clearance", "ground_max_slope"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("max_camera_time_delta", "label_propagation_radius", "label_ttl_sec",
                     "camera_splat_radius", "camera_splat_occlusion_gap", "mask_depth_gap",
                     "mask_depth_gap_ratio"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("max_neighbors", "min_region_voxels", "max_map_voxels", "chunk_size",
                     "camera_max_splat_pixels", "mask_min_layer_points", "max_labels"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.workers, int) or isinstance(self.workers, bool) \
                or not (self.workers == -1 or self.workers >= 1):
            raise ValueError("workers must be -1 (all cores) or a positive integer")
        if not isinstance(self.ground_context_px, int) or isinstance(self.ground_context_px, bool) \
                or self.ground_context_px < 0:
            raise ValueError("ground_context_px must be a nonnegative integer")
        if isinstance(self.ground_surface_labels, str) or not all(
                isinstance(label, str) for label in self.ground_surface_labels):
            raise ValueError("ground_surface_labels must be a list of strings")
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


def _unique_cells(cells, return_index=False, return_inverse=False):
    """``np.unique(cells, axis=0, ...)`` for (N, 3) integer cells: the same sorted
    rows, first-occurrence indices and inverse, through one packed int64 key
    per cell when the cells fit (several times faster on a large cloud)."""
    keys = pack_voxel_keys(cells)
    if keys is None:
        return np.unique(cells, axis=0, return_index=return_index, return_inverse=return_inverse)
    _, index, inverse = np.unique(keys, return_index=True, return_inverse=True)
    unique = np.asarray(cells, dtype=np.int64).reshape(-1, 3)[index]
    extra = ((index,) if return_index else ()) + ((inverse.reshape(-1),) if return_inverse else ())
    return (unique, *extra) if extra else unique


def _unique_pairs(first, second, return_counts=False):
    """``np.unique(np.column_stack([first, second]), axis=0)`` for non-negative
    integer columns (voxel/component/region/label indices), as one 1-D unique of
    ``first * (max(second) + 1) + second`` keys: same rows, order and counts."""
    first, second = np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64)
    base = int(second.max()) + 1 if second.size else 1
    keys = np.unique(first * base + second, return_counts=return_counts)
    pairs = np.column_stack(np.divmod(keys[0] if return_counts else keys, base))
    return (pairs, keys[1]) if return_counts else pairs


def _cell_match(old, new):
    """Indices of identical cells in two lexicographically sorted unique arrays."""
    if not len(old) or not len(new):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    old_keys, new_keys = pack_voxel_keys(old), pack_voxel_keys(new)
    if old_keys is None or new_keys is None:
        old_keys, new_keys = _cell_keys(old), _cell_keys(new)
    index = np.searchsorted(old_keys, new_keys)
    current = np.flatnonzero(index < len(old_keys))
    current = current[old_keys[index[current]] == new_keys[current]]
    return index[current], current


def _kernels(config):
    """The compiled kernels when installed and enabled, else None (NumPy/SciPy)."""
    return native.kernels if config.native_kernels else None


def _surface_normals(points, tree, index, config):
    """PCA normals and a planarity flag for ``points[index]``."""
    kernels = _kernels(config)
    if kernels is not None:
        return kernels.surface_normals(points, index, config.normal_radius, config.max_neighbors, config.workers)
    n = len(points)
    normals = np.zeros((len(index), 3), dtype=np.float64)
    reliable = np.zeros(len(index), dtype=bool)
    k = min(n, config.max_neighbors)
    for start in range(0, len(index), config.chunk_size):
        stop = min(start+config.chunk_size, len(index))
        distances, indices = tree.query(points[index[start:stop]], k=list(range(1, k+1)),
                                        distance_upper_bound=config.normal_radius, workers=config.workers)
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
    return normals, reliable


def _surface_edges(points, tree, normals, reliable, index, config):
    """Kept smooth-surface graph edges whose first endpoint is in ``index``."""
    cosine = math.cos(math.radians(config.normal_angle_deg))
    kernels = _kernels(config)
    if kernels is not None:
        return kernels.surface_edges(points, normals, reliable, index, config.neighbor_radius, config.max_neighbors,
                                     cosine, config.plane_tolerance, config.workers)
    n = len(points)
    k = min(n, config.max_neighbors)
    rows, cols = [np.zeros(0, np.int32)], [np.zeros(0, np.int32)]
    for start in range(0, len(index), config.chunk_size):
        stop = min(start+config.chunk_size, len(index))
        chunk = index[start:stop]
        distance, neighbor = tree.query(points[chunk], k=list(range(1, k+1)),
                                        distance_upper_bound=config.neighbor_radius, workers=config.workers)
        row = np.broadcast_to(chunk[:, None], neighbor.shape)
        valid = np.isfinite(distance) & (neighbor != row)
        a, b = row[valid], neighbor[valid]
        smooth = reliable[a] & reliable[b]
        delta = points[b]-points[a]
        aligned = np.abs(np.einsum("ni,ni->n", normals[a], normals[b])) >= cosine
        flat_a = np.abs(np.einsum("ni,ni->n", normals[a], delta)) <= config.plane_tolerance
        flat_b = np.abs(np.einsum("ni,ni->n", normals[b], delta)) <= config.plane_tolerance
        keep = ~smooth | (aligned & flat_a & flat_b)
        rows.append(a[keep].astype(np.int32)); cols.append(b[keep].astype(np.int32))
    return np.concatenate(rows), np.concatenate(cols)


def _surface_components(n, rows, cols, config):
    kernels = _kernels(config)
    if kernels is not None:
        component = kernels.connected_components(n, rows, cols)
    else:
        graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(n, n)).tocsr()
        _, component = connected_components(graph, directed=False)
    sizes = np.bincount(component)
    keep = np.flatnonzero(sizes >= config.min_region_voxels)
    mapping = np.full(len(sizes), -1, dtype=np.int32)
    mapping[keep] = np.arange(len(keep), dtype=np.int32)
    return mapping[component]


@dataclass
class SurfaceGraph:
    """Cached per-voxel normals and graph edges, aligned with the map's voxel order."""
    points: np.ndarray
    normals: np.ndarray
    reliable: np.ndarray
    rows: np.ndarray
    cols: np.ndarray

    @classmethod
    def empty(cls):
        return cls(np.empty((0, 3)), np.empty((0, 3)), np.empty(0, bool),
                   np.empty(0, np.int32), np.empty(0, np.int32))


def build_surface_graph(points: np.ndarray, config: DenseCloudConfig) -> SurfaceGraph:
    n = len(points)
    if not n:
        return SurfaceGraph.empty()
    tree = cKDTree(points) if _kernels(config) is None else None
    index = np.arange(n)
    normals, reliable = _surface_normals(points, tree, index, config)
    rows, cols = _surface_edges(points, tree, normals, reliable, index, config)
    return SurfaceGraph(points, normals, reliable, rows, cols)


def _within(points, seeds, radius, workers=1, kernels=None):
    """Mask of ``points`` within ``radius`` of any seed position."""
    if not len(seeds) or not len(points):
        return np.zeros(len(points), bool)
    if kernels is not None:
        return kernels.within(points, seeds, radius, workers)
    distance, _ = cKDTree(seeds).query(points, k=1, distance_upper_bound=radius, workers=workers)
    return np.isfinite(distance)


def update_surface_graph(previous: SurfaceGraph, old_index: np.ndarray, new_index: np.ndarray,
                         points: np.ndarray, config: DenseCloudConfig,
                         max_dirty_fraction: float = 0.5) -> tuple[SurfaceGraph, dict]:
    """Recompute normals and edges only near geometry that changed.

    ``old_index[i]`` in ``previous`` is the same voxel as ``new_index[i]`` in
    ``points``. A voxel's normal depends only on points within
    ``normal_radius`` and its edges only on points, normals and neighbour lists
    within ``neighbor_radius``, so everything outside those halos around
    added, moved and removed voxels is reused unchanged. Results equal a full
    rebuild except where k-nearest ties are broken differently.
    """
    n = len(points)
    if not n or not len(previous.points):
        graph = build_surface_graph(points, config)
        return graph, {"incremental": False, "dirty_voxels": n}
    moved = np.any(previous.points[old_index] != points[new_index], axis=1)
    kept_old, kept_new = old_index[~moved], new_index[~moved]
    changed = np.ones(n, bool)
    changed[kept_new] = False
    # int32 like the stored edges (max_map_voxels < 2^31): remapping millions
    # of edges is memory-bound, so half-width indices halve its cost.
    old_to_new = np.full(len(previous.points), -1, np.int32)
    old_to_new[kept_old] = kept_new
    vanished = old_to_new < 0  # removed voxels plus the old position of moved ones
    seeds = np.concatenate([points[changed], previous.points[vanished]])
    kernels = _kernels(config)
    normal_dirty = changed | _within(points, seeds, config.normal_radius, config.workers, kernels)
    edge_dirty = normal_dirty | _within(points, np.concatenate([seeds, points[normal_dirty]]),
                                        config.neighbor_radius, config.workers, kernels)
    if edge_dirty.mean() > max_dirty_fraction:
        graph = build_surface_graph(points, config)
        return graph, {"incremental": False, "dirty_voxels": n}
    tree = cKDTree(points) if kernels is None else None
    normals = np.zeros((n, 3))
    reliable = np.zeros(n, bool)
    normals[kept_new], reliable[kept_new] = previous.normals[kept_old], previous.reliable[kept_old]
    dirty = np.flatnonzero(normal_dirty)
    if len(dirty):
        normals[dirty], reliable[dirty] = _surface_normals(points, tree, dirty, config)
    # Surviving edges, renumbered, except those of dirty voxels (rebuilt below).
    if kernels is not None:
        rows, cols = kernels.remap_edges(previous.rows, previous.cols, old_to_new, edge_dirty)
    else:
        rows = old_to_new[previous.rows]
        cols = old_to_new[previous.cols]
        keep = (rows >= 0) & (cols >= 0)
        keep &= ~edge_dirty[rows]  # a removed row (-1) reads the last voxel, but keep already dropped it
        rows, cols = rows[keep], cols[keep]
    fresh_rows, fresh_cols = _surface_edges(points, tree, normals, reliable, np.flatnonzero(edge_dirty), config)
    graph = SurfaceGraph(points, normals, reliable, np.concatenate([rows, fresh_rows]), np.concatenate([cols, fresh_cols]))
    return graph, {"incremental": True, "dirty_voxels": int(edge_dirty.sum())}


def segment_surfaces(points: np.ndarray, config: DenseCloudConfig) -> np.ndarray:
    """Bounded-neighbor smooth-surface graph, computed in chunks on CPU."""
    n = len(points)
    if not n:
        return np.zeros(0, dtype=np.int32)
    graph = build_surface_graph(points, config)
    return _surface_components(n, graph.rows, graph.cols, config)


def foreground_layer(depths: np.ndarray, config: DenseCloudConfig) -> np.ndarray:
    """Mask of the nearest well-supported depth layer among one mask's points.

    Depths are split at gaps larger than ``max(mask_depth_gap,
    mask_depth_gap_ratio*depth)``. The nearest layer with at least
    ``mask_min_layer_points`` points wins (a lone noise return in front does
    not hide the object); if no layer is that large the nearest one is used.
    """
    keep = np.ones(len(depths), bool)
    if config.mask_depth_gap <= 0 or len(depths) < 2:
        return keep
    order = np.argsort(depths, kind="stable")
    ordered = depths[order]
    gap = np.diff(ordered) > np.maximum(config.mask_depth_gap, config.mask_depth_gap_ratio*ordered[:-1])
    layer = np.r_[0, np.cumsum(gap)]
    sizes = np.bincount(layer)
    supported = np.flatnonzero(sizes >= config.mask_min_layer_points)
    chosen = supported[0] if len(supported) else 0
    keep[order] = layer == chosen
    return keep


def ground_points(world: np.ndarray, u: np.ndarray, v: np.ndarray, mask: np.ndarray, camera_center: np.ndarray,
                  config: DenseCloudConfig) -> np.ndarray | None:
    """Which of the points inside ``mask`` lie on the local ground, or None without one.

    ``world``/``u``/``v`` are the camera-visible points and their pixels. The
    ground is fitted (fit_ground_plane, world z up) to those in the mask's box
    grown by ``ground_context_px``, i.e. including the ground in front of and
    beside the object. A level fallback, or a "ground" that is not below the
    camera (a world frame that is not z up), is not a ground: None.
    """
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not len(rows):
        return None
    m = config.ground_context_px
    context = (u >= cols[0]-m) & (u <= cols[-1]+m) & (v >= rows[0]-m) & (v <= rows[-1]+m)
    plane, fitted = fit_ground_plane(world[context], config.ground_clearance, config.ground_max_slope,
                                     return_fitted=True)
    if not fitted or camera_center[2] <= camera_center[:2] @ plane[:2] + plane[2] + config.ground_clearance:
        return None
    inside = mask[v, u]
    height = world[inside, 2]-world[inside, :2] @ plane[:2]-plane[2]
    return height < config.ground_clearance


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
        self._surface = SurfaceGraph.empty()
        self.last_segmentation = {"incremental": False, "dirty_voxels": 0}

    def _stable_regions(self, components, old_region):
        result = np.full(len(components), -1, dtype=np.int32)
        pairs_valid = (components >= 0) & (old_region >= 0)
        pairs, counts = _unique_pairs(components[pairs_valid], old_region[pairs_valid], return_counts=True)
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
        cells, representative, inverse = _unique_cells(scaled.astype(np.int64), return_index=True, return_inverse=True)
        current_points = points[finite][representative]
        if self.config.input_mode == "scan":
            all_cells = _unique_cells(np.concatenate([self.cells, cells]))
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
        if self.config.incremental_segmentation:
            surface, segmentation = update_surface_graph(self._surface, old, new, new_points, self.config)
        else:
            surface = build_surface_graph(new_points, self.config)
            segmentation = {"incremental": False, "dirty_voxels": len(new_points)}
        components = _surface_components(len(new_points), surface.rows, surface.cols, self.config)
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
        self._surface, self.last_segmentation = surface, segmentation
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
            u = np.round(intr.fx*cam[:, 0]/cam[:, 2]+intr.cx)
            v = np.round(intr.fy*cam[:, 1]/cam[:, 2]+intr.cy)
        # Pixel centres are integer coordinates (as for CameraInfo K/P and the
        # object pipeline): round, not floor, which shifted every label half a
        # pixel down and right, onto the ground below an object's base.
        inside = (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
        valid, cam = valid[inside], cam[inside]
        u, v = u[inside].astype(np.int64), v[inside].astype(np.int64)
        pixel = v*intr.width+u
        depth_buffer = np.full(intr.width*intr.height, np.inf)
        np.minimum.at(depth_buffer, pixel, cam[:, 2])
        visible = cam[:, 2] <= depth_buffer[pixel]+self.config.camera_depth_tolerance
        # Sparse LiDAR leaves holes between foreground returns; a footprint
        # z-buffer keeps background points behind them from looking visible.
        splat = splat_depth_buffer(u, v, cam[:, 2], max(intr.fx, intr.fy), intr.width, intr.height,
                                   self.config.camera_splat_radius, self.config.camera_max_splat_pixels)
        visible &= cam[:, 2] <= splat[pixel]+max(self.config.camera_depth_tolerance,
                                                 self.config.camera_splat_occlusion_gap)
        if camera.depth is not None:
            measured = np.asarray(camera.depth)[v, u]
            visible &= (np.isfinite(measured) & (measured > 0)
                        & (np.abs(cam[:, 2]-measured) <= self.config.camera_depth_tolerance))
        valid, u, v, depth = valid[visible], u[visible], v[visible], cam[visible, 2]
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
                if len(new_labels) >= self.config.max_labels:
                    raise ValueError(f"label vocabulary is full (max_labels={self.config.max_labels})")
                label = len(new_labels)
                new_labels[label] = detection.label
                label_lookup[detection.label] = label
            selected = detection.mask[v, u]
            inside = np.flatnonzero(selected)
            if len(inside) and self.config.ground_exclusion \
                    and detection.label not in self.config.ground_surface_labels:
                ground = ground_points(self._input[valid], u, v, detection.mask, transform[:3, 3], self.config)
                if ground is not None and np.count_nonzero(~ground) >= self.config.mask_min_layer_points:
                    selected[inside[ground]] = False
                    inside = inside[~ground]
            if len(inside):
                # Only the mask's foreground depth layer is the object.
                selected[inside[~foreground_layer(depth[inside], self.config)]] = False
            # Overlapping classes are ambiguous regardless of detection order.
            conflicted |= selected & (point_label != 0) & (point_label != label)
            selected &= (point_label == 0) | (point_label == label)
            point_label[selected] = label
            point_score[selected] = np.maximum(point_score[selected], detection.score)
        point_label[conflicted], point_score[conflicted] = 0, 0
        selected = point_label > 0
        map_index = self._input_map[valid[selected]]
        # A voxel containing conflicting labels remains unresolved this frame.
        unique = _unique_pairs(map_index, point_label[selected])
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
                distance, nearest = cKDTree(self.points[seeds]).query(self.points[target],
                                                                      workers=self.config.workers)
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
                 "segmentation_backend": "python" if _kernels(self.config) is None else "native",
                 "geometry_is_semantic_instances": False}
        return CloudResult(self.stamp, self._input.copy(), valid.copy(), regions_out,
                           semantics_out, confidence_out, source_out, self._visible.copy(),
                           dict(self.labels), self.points.copy(), self.regions.copy(), semantic,
                           confidence, source, stats)
