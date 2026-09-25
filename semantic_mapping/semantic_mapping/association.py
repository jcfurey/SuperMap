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
4. **Relabelling** -- a high-confidence detection whose label the instance has
   not (yet) taken may still be the same object: open-vocabulary labels
   flicker. It joins a visible, unmatched track only when the 2D boxes overlap
   *and* the 3D boxes largely coincide, and Eq. (10) then weighs the new label
   (:func:`associate_relabel`).
5. **Re-identification** -- still-unmatched detections are compared against
   retired instances, in place or by appearance (:func:`reidentify`).
6. Anything left spawns a new tentative object.

Every 2D match (stages 1 and 2) is also validated in 3D (Fig. 2, "2D→3D
Validation"; :func:`split_depth_inconsistent`): image overlap cannot tell a
chair at 2 m from a different chair at 6 m behind it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from semantic_mapping.geometry_utils import bbox3d_gap, centroid, iou_3d, iou_3d_matrix, iou_xyxy, overlap_3d
from semantic_mapping.tracking import TrackKalmanState, mahalanobis_gate
from semantic_mapping.types import ObjectInstance

INVALID_COST = 1e6

DEFAULT_LABEL_MIN_MASS = 0.1
"""Belief mass a label needs before a detection of it may join the instance
(association) or two instances sharing it may merge (object_map)."""


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
    track_label_beliefs: list[dict[str, float]] | None = None,
    detection_labels: list[str] | None = None,
    label_min_mass: float = DEFAULT_LABEL_MIN_MASS,
    label_cost_weight: float = 0.1,
) -> AssociationResult:
    """Hungarian-match predicted track boxes against detection boxes in 2D.

    ``predicted_bboxes[i]`` must correspond to ``tracks[i]`` (already advanced
    through :func:`semantic_mapping.tracking.predict`) so the Mahalanobis gate
    is evaluated against the correct prior covariance. ``candidate_*`` restrict
    which indices take part (for staged association); indices in the result
    are always in the full lists' index space.

    Given ``track_label_beliefs`` and ``detection_labels``, association is
    label-aware: a pair is admissible only when the track's belief holds at
    least ``label_min_mass`` for the detection's label (see
    :func:`labels_compatible`), and the remaining mass shortfall adds
    ``label_cost_weight * (1 - mass)`` to the cost, so between overlapping
    tracks the one that already believes the label wins. Without them the
    match is purely geometric.
    """
    rows = list(range(len(tracks))) if candidate_tracks is None else list(candidate_tracks)
    cols = list(range(len(detection_bboxes))) if candidate_detections is None else list(candidate_detections)
    label_aware = track_label_beliefs is not None and detection_labels is not None

    cost = np.full((len(rows), len(cols)), INVALID_COST, dtype=np.float64)
    for r, i in enumerate(rows):
        for c, j in enumerate(cols):
            label_cost = 0.0
            if label_aware:
                mass = label_mass(detection_labels[j], track_label_beliefs[i])
                if not labels_compatible(detection_labels[j], track_label_beliefs[i], label_min_mass):
                    continue
                label_cost = label_cost_weight * (1.0 - mass)
            iou = iou_xyxy(predicted_bboxes[i], detection_bboxes[j])
            if iou >= iou_threshold:
                cost[r, c] = 1.0 - iou + label_cost

    def accept(i: int, j: int) -> bool:
        return not use_mahalanobis_gate or mahalanobis_gate(tracks[i], detection_bboxes[j])

    return _solve(cost, rows, cols, accept)


def label_mass(label: str, belief: dict[str, float]) -> float:
    return float(belief.get(label, 0.0))


def labels_compatible(label: str, belief: "dict[str, float] | ObjectInstance",
                      min_mass: float = DEFAULT_LABEL_MIN_MASS) -> bool:
    """Whether ``label`` carries at least ``min_mass`` of the belief.

    A label merely *present* in a belief is not enough: a single
    misdetection enters the belief at ~1e-3 mass (Eq. 10 new-label prior),
    and treating that as compatible let one wrong detection license
    cross-class association and merges. ``min_mass = 0`` restores the
    "label present at all" test.
    """
    if isinstance(belief, ObjectInstance):
        belief = belief.label_belief
    mass = belief.get(label)
    return mass is not None and mass >= min_mass


def beliefs_compatible(a: dict[str, float], b: dict[str, float],
                       min_mass: float = DEFAULT_LABEL_MIN_MASS) -> bool:
    """Two beliefs share a label that holds at least ``min_mass`` in both."""
    return any(p >= min_mass and b.get(label, -1.0) >= min_mass for label, p in a.items())


def _labels_compatible(label: str, instance: ObjectInstance, min_mass: float = DEFAULT_LABEL_MIN_MASS) -> bool:
    return labels_compatible(label, instance.label_belief, min_mass)


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
    label_min_mass: float = DEFAULT_LABEL_MIN_MASS,
) -> AssociationResult:
    """Match back-projected detection boxes against existing objects in 3D.

    A pair is admissible when the labels are compatible (the detection's label
    holds at least ``label_min_mass`` of the instance's belief) and either the 3D boxes overlap by
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
            if det_box is None or not labels_compatible(detection_labels[j], obj.label_belief, label_min_mass):
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


def split_depth_inconsistent(
    result: AssociationResult,
    detection_bboxes3d: list[np.ndarray | None],
    objects: list[ObjectInstance],
    max_gap: float,
) -> int:
    """2D→3D validation (Fig. 2): unmatch 2D pairs whose detection lifts away from the instance.

    A pair stays matched when the detection's back-projected box lies within
    ``max_gap`` metres of the instance's box, or when either side has no 3D
    extent to compare (too few depth readings, a 2D-only instance). Split
    pairs return to the unmatched lists, so the detection can re-activate or
    spawn another instance and the instance receives this frame's geometric
    evidence as unmatched. Returns the number of pairs split; ``max_gap <= 0``
    disables the check.
    """
    if max_gap <= 0:
        return 0
    kept, split = [], 0
    for track_idx, det_idx in result.matches:
        det_box, obj = detection_bboxes3d[det_idx], objects[track_idx]
        if det_box is not None and obj.points_world.shape[0] and bbox3d_gap(det_box, obj.bbox3d) > max_gap:
            result.unmatched_tracks.append(track_idx)
            result.unmatched_detections.append(det_idx)
            split += 1
        else:
            kept.append((track_idx, det_idx))
    result.matches = kept
    return split


def associate_relabel(
    tracks: list[TrackKalmanState],
    predicted_bboxes: list[np.ndarray],
    detection_bboxes: list[np.ndarray],
    detection_bboxes3d: list[np.ndarray | None],
    objects: list[ObjectInstance],
    iou_threshold: float = 0.3,
    min_overlap: float = 0.5,
    pad: float = 0.025,
    candidate_tracks: list[int] | None = None,
    candidate_detections: list[int] | None = None,
) -> AssociationResult:
    """Match detections to instances regardless of label, when image and 3D geometry agree.

    Label-gated stages keep a ``person`` in front of a chair out of the chair
    (review 2026-09-24, C2), but they also kept a label the instance had never
    taken from ever reaching Eq. (10), so a flickering detector label split one
    object into co-located instances (paper review 2026-09-25, D1). Here a pair
    is admissible when the 2D boxes overlap by at least ``iou_threshold``, the
    motion gate accepts it, and at least ``min_overlap`` of the smaller 3D box
    (each grown by ``pad``) lies inside the other: a person standing in front
    of the chair lifts to a box in front of it and stays out. Both sides need
    3D extent; ``min_overlap <= 0`` disables the stage.
    """
    rows = list(range(len(objects))) if candidate_tracks is None else list(candidate_tracks)
    cols = list(range(len(detection_bboxes))) if candidate_detections is None else list(candidate_detections)
    cost = np.full((len(rows), len(cols)), INVALID_COST, dtype=np.float64)
    if min_overlap > 0:
        for r, i in enumerate(rows):
            if objects[i].points_world.shape[0] == 0:
                continue
            for c, j in enumerate(cols):
                if detection_bboxes3d[j] is None:
                    continue
                iou = iou_xyxy(predicted_bboxes[i], detection_bboxes[j])
                if iou >= iou_threshold and overlap_3d(detection_bboxes3d[j], objects[i].bbox3d, pad) >= min_overlap:
                    cost[r, c] = 1.0 - iou
    return _solve(cost, rows, cols, lambda i, j: mahalanobis_gate(tracks[i], detection_bboxes[j]))


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
    label_min_mass: float = DEFAULT_LABEL_MIN_MASS,
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

    The pair tests are evaluated as arrays over (retired x detections): with
    up to ``max_retired_instances`` identities kept, a per-pair Python loop
    dominated the frame (doc/review-2026-09-24.md, P1). Age and label
    compatibility prefilter the rows, then 3D IoU, both containment tests,
    and one matrix product of unit embeddings score every surviving pair.
    """
    rows = list(range(len(retired)))
    cols = list(range(len(detection_bboxes3d))) if candidate_detections is None else list(candidate_detections)
    cost = np.full((len(rows), len(cols)), INVALID_COST, dtype=np.float64)
    by_place: set[tuple[int, int]] = set()
    col_idx = np.array([c for c, j in enumerate(cols) if detection_bboxes3d[j] is not None], dtype=np.int64)
    if not rows or col_idx.size == 0:
        return _solve(cost, rows, cols, lambda i, j: True), by_place

    # Admissible (row, col) pairs before any geometry: age and label mass.
    admissible = np.zeros((len(rows), len(cols)), dtype=bool)
    if max_age_sec > 0:
        fresh = np.array([now - obj.latest_stamp <= max_age_sec for obj in retired])
    else:
        fresh = np.ones(len(rows), dtype=bool)
    for label in {detection_labels[cols[c]] for c in col_idx}:
        label_cols = col_idx[[detection_labels[cols[c]] == label for c in col_idx]]
        compatible = fresh & np.array([labels_compatible(label, obj.label_belief, label_min_mass) for obj in retired])
        admissible[np.ix_(compatible, label_cols)] = True
    row_idx = np.flatnonzero(admissible.any(axis=1))
    if row_idx.size == 0:
        return _solve(cost, rows, cols, lambda i, j: True), by_place
    admissible = admissible[np.ix_(row_idx, col_idx)]

    obj_boxes = np.array([retired[r].bbox3d for r in row_idx], dtype=np.float64).reshape(-1, 6)
    det_boxes = np.array([detection_bboxes3d[cols[c]] for c in col_idx], dtype=np.float64).reshape(-1, 6)
    obj_centers = (obj_boxes[:, :3] + obj_boxes[:, 3:]) / 2.0
    det_centers = (det_boxes[:, :3] + det_boxes[:, 3:]) / 2.0
    has_box = np.any(obj_boxes[:, 3:] > obj_boxes[:, :3], axis=1)

    def inside(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """(P, B): whether each point lies inside each box expanded by the margin."""
        p = points[:, None, :]
        return np.all((p >= boxes[None, :, :3] - containment_margin) & (p <= boxes[None, :, 3:] + containment_margin), axis=2)

    same_place = has_box[:, None] & (
        (iou_3d_matrix(obj_boxes, det_boxes) > iou_threshold)
        | inside(det_centers, obj_boxes).T
        | inside(obj_centers, det_boxes)
    )

    # Cosine similarity where both sides carry an embedding; descriptors of
    # different dimension are dissimilar (appearance.cosine_similarity).
    has_similarity = np.zeros_like(admissible)
    similarity = np.zeros(admissible.shape, dtype=np.float64)
    obj_embeddings = [retired[r].embedding for r in row_idx]
    det_embeddings = [detection_embeddings[cols[c]] for c in col_idx]
    obj_with = np.array([e is not None for e in obj_embeddings])
    det_with = np.array([e is not None for e in det_embeddings])
    has_similarity[np.ix_(obj_with, det_with)] = True
    for dim in {np.asarray(e).size for e in det_embeddings if e is not None}:
        r_sel = np.flatnonzero([e is not None and np.asarray(e).size == dim for e in obj_embeddings])
        c_sel = np.flatnonzero([e is not None and np.asarray(e).size == dim for e in det_embeddings])
        if r_sel.size == 0:
            continue
        a = np.array([np.asarray(obj_embeddings[r], dtype=np.float64).ravel() for r in r_sel])
        b = np.array([np.asarray(det_embeddings[c], dtype=np.float64).ravel() for c in c_sel])
        a_norm, b_norm = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        denominator = a_norm[:, None] * b_norm[None, :]
        dot = a @ b.T
        similarity[np.ix_(r_sel, c_sel)] = np.where(denominator > 0, dot / np.where(denominator > 0, denominator, 1.0), 0.0)

    valid = admissible & ~(has_similarity & (similarity < min_similarity))  # a different object, wherever it is
    valid &= same_place | has_similarity  # a relocation can only be claimed on appearance
    appearance_cost = np.where(has_similarity, 1.0 - similarity, 0.5)
    distance = np.linalg.norm(det_centers[None, :, :] - obj_centers[:, None, :], axis=2)
    pair_cost = appearance_cost - np.where(same_place, 0.5, 0.0) + 0.01 * distance
    sub = cost[row_idx]
    sub[:, col_idx] = np.where(valid, pair_cost, INVALID_COST)
    cost[row_idx] = sub
    for r, c in zip(*np.nonzero(valid & same_place)):
        by_place.add((int(row_idx[r]), cols[int(col_idx[c])]))

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
