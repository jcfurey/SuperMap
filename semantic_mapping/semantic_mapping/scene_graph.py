"""Spatio-temporal scene graph construction (Sec. IV-C).

The map M_t is abstracted into a graph G = (V, E_s, E_t): nodes V are object
instances, spatial edges E_s are class-dependent geometric predicates
(on/under/beside) evaluated between nearby objects, and temporal edges E_t
trace each instance's own trajectory over time via the association result.
For real-time operation, a spatial index selects pairs of nearby boxes so
predicates are evaluated without losing neighbors across cluster boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from semantic_mapping.geometry_utils import iou_xy
from semantic_mapping.types import ObjectInstance, ObjectStatus

DEFAULT_MAX_TRAJECTORY_LENGTH = 200


@dataclass
class SpatialEdge:
    subject_id: int
    predicate: str
    object_id: int


@dataclass
class SceneGraph:
    node_ids: list[int] = field(default_factory=list)
    spatial_edges: list[SpatialEdge] = field(default_factory=list)


def _neighbor_pairs(objects: list[ObjectInstance], radius: float) -> list[tuple[int, int]]:
    """Every pair whose boxes come within ``radius`` of each other, once, in stable order.

    The distance is the gap between the axis-aligned boxes, not between
    their centroids: a mug at the end of a 4 m table has its centroid more
    than 2 m from the table's, yet sits on it. The gap never exceeds the
    centroid distance, so every pair the centroid test found is still
    found. Candidates come from a KD-tree on centroids with a per-object
    radius widened by both half-diagonals (a lower bound on the gap), and
    the exact box gap decides.
    """
    if len(objects) < 2:
        return []
    boxes = np.array([obj.bbox3d for obj in objects], dtype=np.float64).reshape(-1, 6)
    centers = (boxes[:, :3] + boxes[:, 3:]) / 2.0
    half_diagonals = np.linalg.norm(np.clip(boxes[:, 3:] - boxes[:, :3], 0.0, None), axis=1) / 2.0
    tree = cKDTree(centers)
    max_half = float(half_diagonals.max())
    pairs: list[tuple[int, int]] = []
    for i, candidates in enumerate(tree.query_ball_point(centers, r=radius + half_diagonals + max_half)):
        others = np.array([j for j in candidates if j > i], dtype=np.int64)
        if others.size == 0:
            continue
        gaps = np.clip(np.maximum(boxes[others, :3] - boxes[i, 3:], boxes[i, :3] - boxes[others, 3:]), 0.0, None)
        close = np.linalg.norm(gaps, axis=1) <= radius
        pairs.extend((i, int(j)) for j in others[close])
    return sorted(pairs)


def footprint_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of box ``a``'s XY footprint that lies over box ``b``'s."""
    overlap = np.clip(np.minimum(a[3:5], b[3:5]) - np.maximum(a[0:2], b[0:2]), 0.0, None)
    area = float(np.prod(np.clip(a[3:5] - a[0:2], 0.0, None)))
    return float(np.prod(overlap)) / area if area > 0 else 0.0


def _on_predicate(a: ObjectInstance, b: ObjectInstance, z_tolerance: float, xy_iou_threshold: float,
                  min_footprint_fraction: float = 0.0) -> bool:
    """On(A, B) <=> (z_A_min ~= z_B_max) AND (IoU_xy(B_A, B_B) > gamma).

    The IoU term alone rejects a small object on a large support -- a mug on
    a 4 x 1 m table has IoU 0.0025 -- although the paper's own examples are
    objects on tables. With ``min_footprint_fraction > 0`` the XY condition
    also holds when at least that fraction of A's footprint lies over B
    (doc/paper-review-2026-09-25.md, D12); 0 keeps the IoU test alone.
    """
    z_a_min = a.bbox3d[2]
    z_b_max = b.bbox3d[5]
    if abs(z_a_min - z_b_max) > z_tolerance:
        return False
    if iou_xy(a.bbox3d, b.bbox3d) > xy_iou_threshold:
        return True
    return min_footprint_fraction > 0 and footprint_fraction(a.bbox3d, b.bbox3d) >= min_footprint_fraction


def _beside_predicate(a: ObjectInstance, b: ObjectInstance, z_tolerance: float,
                       xy_iou_threshold: float, beside_max_distance: float) -> bool:
    """Beside(A, B): comparable support height, negligible footprint overlap, and
    horizontally close centroids -- the complement of a stacking relation.
    """
    z_a_min, z_b_min = a.bbox3d[2], b.bbox3d[2]
    same_support_level = abs(z_a_min - z_b_min) <= z_tolerance
    barely_overlaps = iou_xy(a.bbox3d, b.bbox3d) <= xy_iou_threshold
    horizontal_distance = float(np.linalg.norm(a.center[:2] - b.center[:2]))
    return same_support_level and barely_overlaps and horizontal_distance <= beside_max_distance


DEFAULT_SUPPORT_CLASSES: tuple[str, ...] = (
    "table", "desk", "shelf", "counter", "countertop", "cabinet", "dresser", "nightstand",
    "bed", "sofa", "couch", "bench", "stool", "chair", "cart", "box", "floor",
)
"""Classes that can carry another object. The ``On`` predicate is class-dependent
(Sec. IV-C): geometry alone would also say a table sits "on" a rug or a wall
"on" the floor, so the supporting object must be something that plausibly
supports."""


class SpatialEdgeCache:
    """Remembers each neighboring pair's edges keyed by the predicate inputs.

    Objects that were not touched this frame keep the same box and label, so
    a pair whose members are all unchanged yields exactly the edges it
    yielded last time. Only pairs with changed predicate inputs need their
    geometric relations evaluated again.
    """

    def __init__(self) -> None:
        self._edges: dict[tuple, list[SpatialEdge]] = {}

    @staticmethod
    def signature(objects: list[ObjectInstance], members: tuple[int, int]) -> tuple:
        return tuple(
            (objects[i].instance_id, objects[i].label, tuple(objects[i].bbox3d.tolist())) for i in members
        )

    def get(self, key: tuple) -> list[SpatialEdge] | None:
        return self._edges.get(key)

    def put(self, key: tuple, edges: list[SpatialEdge]) -> None:
        self._edges[key] = edges

    def retain(self, keys: set[tuple]) -> None:
        """Drop entries for pairs that no longer exist."""
        self._edges = {k: v for k, v in self._edges.items() if k in keys}


def _pair_edges(a, b, z_tolerance, xy_iou_threshold, beside_max_distance, supports,
                on_min_footprint_fraction=0.0) -> list[SpatialEdge]:
    edges: list[SpatialEdge] = []
    for subject, support in ((a, b), (b, a)):
        can_support = supports is None or support.label in supports
        if can_support and _on_predicate(subject, support, z_tolerance, xy_iou_threshold, on_min_footprint_fraction):
            edges.append(SpatialEdge(subject.instance_id, "on", support.instance_id))
            edges.append(SpatialEdge(support.instance_id, "under", subject.instance_id))
    if not edges and _beside_predicate(a, b, z_tolerance, xy_iou_threshold, beside_max_distance):
        edges.append(SpatialEdge(a.instance_id, "beside", b.instance_id))
        edges.append(SpatialEdge(b.instance_id, "beside", a.instance_id))
    return edges


def build_spatial_edges(
    objects: list[ObjectInstance],
    cluster_radius: float = 2.0,
    z_tolerance: float = 0.1,
    xy_iou_threshold: float = 0.05,
    beside_max_distance: float = 1.0,
    support_classes: tuple[str, ...] | list[str] | None = DEFAULT_SUPPORT_CLASSES,
    cache: SpatialEdgeCache | None = None,
    on_min_footprint_fraction: float = 0.5,
) -> list[SpatialEdge]:
    """Evaluate predicates for every pair within ``cluster_radius`` of each other.

    Emits ``on`` (A on B, with B's class in ``support_classes``; pass an empty
    collection to make it purely geometric), its inverse ``under`` (B under
    A), and symmetric ``beside`` edges. With a ``cache``, pairs whose
    members are unchanged since the last call reuse their edges.
    """
    edges: list[SpatialEdge] = []
    objects = sorted(objects, key=lambda obj: obj.instance_id)
    pairs = _neighbor_pairs(objects, cluster_radius)
    supports = set(support_classes) if support_classes else None
    keys: set[tuple] = set()
    parameters = (z_tolerance, xy_iou_threshold, beside_max_distance, on_min_footprint_fraction,
                  tuple(sorted(supports)) if supports is not None else None)

    for i, j in pairs:
        key = (parameters, SpatialEdgeCache.signature(objects, (i, j))) if cache is not None else None
        pair_edges = cache.get(key) if cache is not None else None
        if pair_edges is None:
            pair_edges = _pair_edges(objects[i], objects[j], z_tolerance, xy_iou_threshold, beside_max_distance,
                                     supports, on_min_footprint_fraction)
            if cache is not None:
                cache.put(key, pair_edges)
        if cache is not None:
            keys.add(key)
        edges.extend(pair_edges)

    if cache is not None:
        cache.retain(keys)
    return edges


def record_trajectory_sample(
    instance: ObjectInstance,
    stamp: float,
    max_length: int = DEFAULT_MAX_TRAJECTORY_LENGTH,
    min_motion: float = 0.05,
) -> None:
    """Append a temporal-edge sample (Sec. IV-C E_t) for this instance's trajectory.

    Only records a new waypoint when the object has moved appreciably or its
    status changed, keeping long-lived static objects from bloating history.
    """
    sample = (stamp, instance.center.copy(), instance.status.value)
    if instance.trajectory:
        last_stamp, last_center, last_status = instance.trajectory[-1]
        moved = float(np.linalg.norm(instance.center - last_center)) >= min_motion
        status_changed = last_status != instance.status.value
        if not moved and not status_changed:
            return
    instance.trajectory.append(sample)
    if len(instance.trajectory) > max_length:
        del instance.trajectory[: len(instance.trajectory) - max_length]


def build_scene_graph(
    objects: list[ObjectInstance],
    cluster_radius: float = 2.0,
    z_tolerance: float = 0.1,
    xy_iou_threshold: float = 0.05,
    beside_max_distance: float = 1.0,
    node_statuses: tuple[ObjectStatus, ...] = (ObjectStatus.ACTIVE, ObjectStatus.OCCLUDED, ObjectStatus.DISAPPEARED),
    edge_statuses: tuple[ObjectStatus, ...] = (ObjectStatus.ACTIVE, ObjectStatus.OCCLUDED),
    support_classes: tuple[str, ...] | list[str] | None = DEFAULT_SUPPORT_CLASSES,
    cache: SpatialEdgeCache | None = None,
    on_min_footprint_fraction: float = 0.5,
) -> SceneGraph:
    """Build G = (V, E_s, E_t) from the current map state.

    ``node_statuses`` controls which instances are queryable at all (kept
    broad, e.g. including DISAPPEARED, to support "recall past scenes"
    queries over an object's trajectory); ``edge_statuses`` restricts
    geometric predicate evaluation to instances with a currently-meaningful
    3D position. Temporal edges are not materialized as a separate list
    here: they are implicit in each node's ``trajectory`` field (populated
    incrementally by :func:`record_trajectory_sample`), which the
    serialization layer reads directly.
    """
    # A track that never acquired depth has no spatial location, including
    # after it expires. Retired mapped objects retain their last known box.
    nodes = [obj for obj in objects if obj.status in node_statuses
             and np.any(obj.bbox3d[3:] > obj.bbox3d[:3])]
    edge_eligible = [obj for obj in nodes if obj.status in edge_statuses]
    spatial_edges = build_spatial_edges(
        edge_eligible, cluster_radius, z_tolerance, xy_iou_threshold, beside_max_distance, support_classes,
        cache=cache, on_min_footprint_fraction=on_min_footprint_fraction,
    )
    return SceneGraph(node_ids=[obj.instance_id for obj in nodes], spatial_edges=spatial_edges)
