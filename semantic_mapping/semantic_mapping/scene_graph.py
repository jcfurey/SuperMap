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


DEFAULT_SUPPORT_CLASSES: tuple[str, ...] = (
    "table", "desk", "shelf", "counter", "countertop", "cabinet", "dresser", "nightstand",
    "bed", "sofa", "couch", "bench", "stool", "chair", "cart", "box", "floor",
)
"""Classes that can carry another object. The ``On`` predicate is class-dependent
(Sec. IV-C): geometry alone would also say a table sits "on" a rug or a wall
"on" the floor, so the supporting object must be something that plausibly
supports."""


def _footprint_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise :func:`~semantic_mapping.geometry_utils.iou_xy` of two (P, 6) box arrays,
    with the same operations in the same order, hence the same values."""
    inter_w = np.maximum(0.0, np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 0], b[:, 0]))
    inter_h = np.maximum(0.0, np.minimum(a[:, 4], b[:, 4]) - np.maximum(a[:, 1], b[:, 1]))
    inter = inter_w * inter_h
    area_a = np.maximum(0.0, a[:, 3] - a[:, 0]) * np.maximum(0.0, a[:, 4] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 3] - b[:, 0]) * np.maximum(0.0, b[:, 4] - b[:, 1])
    union = area_a + area_b - inter
    return np.divide(inter, union, out=np.zeros_like(union), where=union > 1e-12)


def _footprint_fractions(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise :func:`footprint_fraction` of two (P, 6) box arrays."""
    overlap = np.clip(np.minimum(a[:, 3:5], b[:, 3:5]) - np.maximum(a[:, 0:2], b[:, 0:2]), 0.0, None)
    extent = np.clip(a[:, 3:5] - a[:, 0:2], 0.0, None)
    area = extent[:, 0] * extent[:, 1]
    return np.divide(overlap[:, 0] * overlap[:, 1], area, out=np.zeros_like(area), where=area > 0)


def _horizontal_within(a: np.ndarray, b: np.ndarray, limit: float) -> np.ndarray:
    """Whether the XY distance between box centres is at most ``limit``, row-wise.

    The scalar test took ``np.linalg.norm`` of the 2-vector, a BLAS dot
    product whose rounding can differ from the elementwise sum by one unit
    in the last place. Pairs that close to the limit are decided by that
    same scalar expression, so the outcome never changes.
    """
    difference = (a[:, :2] + a[:, 3:5]) / 2.0 - (b[:, :2] + b[:, 3:5]) / 2.0
    distance = np.sqrt(difference[:, 0] * difference[:, 0] + difference[:, 1] * difference[:, 1])
    within = distance <= limit
    for k in np.flatnonzero(np.abs(distance - limit) <= 1e-9 * max(abs(limit), 1.0)):
        within[k] = float(np.linalg.norm(difference[k])) <= limit
    return within


def _pair_relations(boxes: np.ndarray, can_support: np.ndarray, first: np.ndarray, second: np.ndarray,
                    z_tolerance: float, xy_iou_threshold: float, beside_max_distance: float,
                    on_min_footprint_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``on`` in each direction and ``beside`` for every pair ``(first[k], second[k])``.

    On(A, B) <=> (z_A_min ~= z_B_max) AND (IoU_xy(B_A, B_B) > gamma), with B
    of a supporting class. The IoU term alone rejects a small object on a
    large support -- a mug on a 4 x 1 m table has IoU 0.0025 -- although the
    paper's own examples are objects on tables. With
    ``on_min_footprint_fraction > 0`` the XY condition also holds when at
    least that fraction of A's footprint lies over B
    (doc/paper-review-2026-09-25.md, D12); 0 keeps the IoU test alone.

    Beside(A, B) holds for pairs with neither ``on`` relation: comparable
    support height, negligible footprint overlap, and horizontally close
    centroids -- the complement of a stacking relation.
    """
    a, b = boxes[first], boxes[second]
    iou = _footprint_iou(a, b)  # symmetric: IoU_xy(A, B) == IoU_xy(B, A) exactly
    overlapping = iou > xy_iou_threshold

    def on(subject, support, support_ok):
        stacked = support_ok & ~(np.abs(subject[:, 2] - support[:, 5]) > z_tolerance)
        if on_min_footprint_fraction > 0:
            over = _footprint_fractions(subject, support) >= on_min_footprint_fraction
            return stacked & (overlapping | over)
        return stacked & overlapping

    on_first = on(a, b, can_support[second])
    on_second = on(b, a, can_support[first])
    beside = ~on_first & ~on_second & (np.abs(a[:, 2] - b[:, 2]) <= z_tolerance) & (iou <= xy_iou_threshold)
    candidates = np.flatnonzero(beside)
    beside[candidates] = _horizontal_within(a[candidates], b[candidates], beside_max_distance)
    return on_first, on_second, beside


def build_spatial_edges(
    objects: list[ObjectInstance],
    cluster_radius: float = 2.0,
    z_tolerance: float = 0.1,
    xy_iou_threshold: float = 0.05,
    beside_max_distance: float = 1.0,
    support_classes: tuple[str, ...] | list[str] | None = DEFAULT_SUPPORT_CLASSES,
    on_min_footprint_fraction: float = 0.5,
) -> list[SpatialEdge]:
    """Evaluate predicates for every pair within ``cluster_radius`` of each other.

    Emits ``on`` (A on B, with B's class in ``support_classes``; pass an empty
    collection to make it purely geometric), its inverse ``under`` (B under
    A), and symmetric ``beside`` edges. All pairs are evaluated together in
    array operations (:func:`_pair_relations`), so the cost per pair is a
    few arithmetic operations rather than a Python call per predicate.
    """
    objects = sorted(objects, key=lambda obj: obj.instance_id)
    pairs = _neighbor_pairs(objects, cluster_radius)
    if not pairs:
        return []
    supports = set(support_classes) if support_classes else None
    boxes = np.array([obj.bbox3d for obj in objects], dtype=np.float64).reshape(-1, 6)
    can_support = np.array([supports is None or obj.label in supports for obj in objects], dtype=bool)
    first, second = np.array(pairs, dtype=np.int64).T
    on_first, on_second, beside = _pair_relations(
        boxes, can_support, first, second, z_tolerance, xy_iou_threshold, beside_max_distance,
        on_min_footprint_fraction)

    ids = [obj.instance_id for obj in objects]
    edges: list[SpatialEdge] = []
    for k in np.flatnonzero(on_first | on_second | beside).tolist():
        a, b = ids[first[k]], ids[second[k]]
        if on_first[k]:
            edges += [SpatialEdge(a, "on", b), SpatialEdge(b, "under", a)]
        if on_second[k]:
            edges += [SpatialEdge(b, "on", a), SpatialEdge(a, "under", b)]
        if beside[k]:
            edges += [SpatialEdge(a, "beside", b), SpatialEdge(b, "beside", a)]
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
        on_min_footprint_fraction=on_min_footprint_fraction,
    )
    return SceneGraph(node_ids=[obj.instance_id for obj in nodes], spatial_edges=spatial_edges)
