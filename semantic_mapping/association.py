"""2D-3D instance association and re-activation (Sec. IV-B).

Solves P(I_t | M_t-1, Q_t): assigning each 2D detection a stable instance ID.
The pipeline runs association in stages:

1. **2D, high-confidence detections** -- Hungarian assignment over IoU between
   detections and track boxes predicted by the motion-compensated projection
   in :mod:`semantic_mapping.tracking`, gated by a Mahalanobis check.
2. **2D, low-confidence detections** (ByteTrack's second association, "associate
   every detection box") -- leftover tracks get a chance to match the
   low-score detections at a looser IoU threshold, so a briefly-occluded or
   blurred object keeps its track instead of dropping it. Low-score
   detections that still don't match never spawn new objects.
3. **3D-aware re-activation** -- a high-confidence detection that found no 2D
   match is compared *in 3D* (back-projected box vs. existing object boxes,
   label-compatible) against the still-unmatched objects, including occluded
   and tentative ones. This is what keeps one identity when the 2D prediction
   is off after a long occlusion or an aggressive viewpoint change, instead of
   fragmenting the object into a chain of new IDs.
4. Anything left spawns a new tentative object.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from semantic_mapping.geometry_utils import centroid, iou_3d, iou_xyxy
from semantic_mapping.tracking import TrackKalmanState, mahalanobis_gate
from semantic_mapping.types import ObjectInstance

INVALID_COST = 1e6


@dataclass
class AssociationResult:
    matches: list[tuple[int, int]] = field(default_factory=list)
    """List of (track_index, detection_index) pairs, in the caller's index space."""

    unmatched_tracks: list[int] = field(default_factory=list)
    unmatched_detections: list[int] = field(default_factory=list)


def _solve(cost: np.ndarray, rows: list[int], cols: list[int], accept) -> AssociationResult:
    """Hungarian-solve a (len(rows) x len(cols)) cost matrix and map back to caller indices."""
    result = AssociationResult()
    if not rows or not cols:
        result.unmatched_tracks = list(rows)
        result.unmatched_detections = list(cols)
        return result

    row_idx, col_idx = linear_sum_assignment(cost)
    matched_rows, matched_cols = set(), set()
    for r, c in zip(row_idx, col_idx):
        if cost[r, c] >= INVALID_COST or not accept(rows[r], cols[c]):
            continue
        result.matches.append((rows[r], cols[c]))
        matched_rows.add(r)
        matched_cols.add(c)

    result.unmatched_tracks = [rows[r] for r in range(len(rows)) if r not in matched_rows]
    result.unmatched_detections = [cols[c] for c in range(len(cols)) if c not in matched_cols]
    return result


def associate(
    tracks: list[TrackKalmanState],
    predicted_bboxes: list[np.ndarray],
    detection_bboxes: list[np.ndarray],
    iou_threshold: float = 0.3,
    use_mahalanobis_gate: bool = True,
    candidate_tracks: list[int] | None = None,
    candidate_detections: list[int] | None = None,
) -> AssociationResult:
    """Hungarian-match predicted track boxes against detection boxes in 2D.

    ``predicted_bboxes[i]`` must correspond to ``tracks[i]`` (already advanced
    through :func:`semantic_mapping.tracking.predict`) so the Mahalanobis gate
    is evaluated against the correct prior covariance. ``candidate_*`` restrict
    which indices take part (for staged association); indices in the result
    are always in the full lists' index space.
    """
    rows = list(range(len(tracks))) if candidate_tracks is None else list(candidate_tracks)
    cols = list(range(len(detection_bboxes))) if candidate_detections is None else list(candidate_detections)

    cost = np.full((len(rows), len(cols)), INVALID_COST, dtype=np.float64)
    for r, i in enumerate(rows):
        for c, j in enumerate(cols):
            iou = iou_xyxy(predicted_bboxes[i], detection_bboxes[j])
            if iou >= iou_threshold:
                cost[r, c] = 1.0 - iou

    def accept(i: int, j: int) -> bool:
        return not use_mahalanobis_gate or mahalanobis_gate(tracks[i], detection_bboxes[j])

    return _solve(cost, rows, cols, accept)


def _labels_compatible(label: str, instance: ObjectInstance) -> bool:
    return label in instance.label_belief


def _inside_expanded(point: np.ndarray, bbox3d: np.ndarray, margin: float) -> bool:
    return bool(np.all(point >= bbox3d[:3] - margin) and np.all(point <= bbox3d[3:] + margin))


def associate_3d(
    detection_bboxes3d: list[np.ndarray | None],
    detection_labels: list[str],
    objects: list[ObjectInstance],
    iou_threshold: float = 0.05,
    containment_margin: float = 0.25,
    candidate_objects: list[int] | None = None,
    candidate_detections: list[int] | None = None,
) -> AssociationResult:
    """Match back-projected detection boxes against existing objects in 3D.

    A pair is admissible when the labels are compatible (the detection's label
    is among the instance's label belief) and either the 3D boxes overlap by
    more than ``iou_threshold`` or one box's centroid lies inside the other
    expanded by ``containment_margin`` meters -- the containment test is what
    lets a small partial view of a large object (a corner of a sofa) still
    re-attach to it despite a tiny IoU. Detections with no 3D box (``None``)
    never match.
    """
    rows = list(range(len(objects))) if candidate_objects is None else list(candidate_objects)
    cols = list(range(len(detection_bboxes3d))) if candidate_detections is None else list(candidate_detections)

    cost = np.full((len(rows), len(cols)), INVALID_COST, dtype=np.float64)
    for r, i in enumerate(rows):
        obj = objects[i]
        if obj.points_world.shape[0] == 0:
            continue
        for c, j in enumerate(cols):
            det_box = detection_bboxes3d[j]
            if det_box is None or not _labels_compatible(detection_labels[j], obj):
                continue
            iou = iou_3d(det_box, obj.bbox3d)
            det_center = centroid(det_box)
            contained = (
                _inside_expanded(det_center, obj.bbox3d, containment_margin)
                or _inside_expanded(obj.center, det_box, containment_margin)
            )
            if iou > iou_threshold or contained:
                distance = float(np.linalg.norm(det_center - obj.center))
                cost[r, c] = (1.0 - iou) + 0.01 * distance

    return _solve(cost, rows, cols, lambda i, j: True)


def reidentify(
    detection_bboxes3d: list[np.ndarray | None],
    detection_labels: list[str],
    detection_embeddings: list[np.ndarray | None],
    retired: list[ObjectInstance],
    min_similarity: float = 0.85,
    iou_threshold: float = 0.05,
    containment_margin: float = 0.25,
    max_age_sec: float = 0.0,
    now: float = 0.0,
    candidate_detections: list[int] | None = None,
) -> tuple[AssociationResult, set[tuple[int, int]]]:
    """Match still-unmatched detections against retired (disappeared) instances.

    An identity comes back under its old ID in two situations (Sec. IV-B,
    "stable identities across relocations"): the object is detected again
    where it used to be (label compatible, 3D box overlapping or containing
    the old one), or it is detected somewhere else looking the same
    (label compatible, appearance similarity at least ``min_similarity``).
    When both sides carry an embedding, a similarity below the threshold
    vetoes even a same-place match, so a different object put in the old
    spot gets a new ID. ``max_age_sec`` (0 = unlimited) bounds how long ago
    the instance was last seen. Returns the assignment and the set of
    (instance index, detection index) pairs that matched by place.
    """
    from semantic_mapping.appearance import cosine_similarity

    rows = list(range(len(retired)))
    cols = list(range(len(detection_bboxes3d))) if candidate_detections is None else list(candidate_detections)
    cost = np.full((len(rows), len(cols)), INVALID_COST, dtype=np.float64)
    by_place: set[tuple[int, int]] = set()
    for r, i in enumerate(rows):
        obj = retired[i]
        if max_age_sec > 0 and now - obj.latest_stamp > max_age_sec:
            continue
        has_box = bool(np.any(obj.bbox3d[3:] > obj.bbox3d[:3]))
        for c, j in enumerate(cols):
            det_box = detection_bboxes3d[j]
            if det_box is None or not _labels_compatible(detection_labels[j], obj):
                continue
            det_center = centroid(det_box)
            same_place = has_box and (
                iou_3d(det_box, obj.bbox3d) > iou_threshold
                or _inside_expanded(det_center, obj.bbox3d, containment_margin)
                or _inside_expanded(obj.center, det_box, containment_margin)
            )
            similarity = None
            if detection_embeddings[j] is not None and obj.embedding is not None:
                similarity = cosine_similarity(detection_embeddings[j], obj.embedding)
                if similarity < min_similarity:
                    continue  # looks like a different object, wherever it is
            if not same_place and similarity is None:
                continue  # a relocation can only be claimed on appearance
            appearance_cost = (1.0 - similarity) if similarity is not None else 0.5
            cost[r, c] = appearance_cost - (0.5 if same_place else 0.0) + 0.01 * float(np.linalg.norm(det_center - obj.center))
            if same_place:
                by_place.add((i, j))
    result = _solve(cost, rows, cols, lambda i, j: True)
    return result, {pair for pair in by_place if pair in set(result.matches)}


def should_reactivate(
    frames_since_seen: int,
    max_occlusion_frames: int,
) -> bool:
    """Whether a dormant track is still within its re-activation window.

    Tracks within the window remain eligible for future association
    (Sec. IV-B.1 identity preservation through occlusion); tracks beyond it
    are handed off to the geometric-consistency module to be confirmed
    disappeared rather than matched again.
    """
    return frames_since_seen <= max_occlusion_frames
