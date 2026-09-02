"""2D-3D instance association and re-activation (Sec. IV-B).

Solves P(I_t | M_t-1, Q_t): assigning each 2D detection a stable instance ID
by matching it against tracks whose predicted position comes from the
motion-compensated projection in :mod:`semantic_mapping.tracking`. Matching
uses Hungarian assignment over an IoU cost, gated by a Mahalanobis distance
check so implausible matches (e.g. after long occlusions) fall through to
either re-activating a dormant track or spawning a new one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from semantic_mapping.geometry_utils import iou_xyxy
from semantic_mapping.tracking import TrackKalmanState, mahalanobis_gate

INVALID_COST = 1e6


@dataclass
class AssociationResult:
    matches: list[tuple[int, int]] = field(default_factory=list)
    """List of (track_index, detection_index) pairs."""

    unmatched_tracks: list[int] = field(default_factory=list)
    unmatched_detections: list[int] = field(default_factory=list)


def _cost_matrix(
    predicted_bboxes: list[np.ndarray],
    detection_bboxes: list[np.ndarray],
    iou_threshold: float,
) -> np.ndarray:
    n_tracks, n_dets = len(predicted_bboxes), len(detection_bboxes)
    cost = np.full((n_tracks, n_dets), INVALID_COST, dtype=np.float64)
    for i, pred_box in enumerate(predicted_bboxes):
        for j, det_box in enumerate(detection_bboxes):
            iou = iou_xyxy(pred_box, det_box)
            if iou >= iou_threshold:
                cost[i, j] = 1.0 - iou
    return cost


def associate(
    tracks: list[TrackKalmanState],
    predicted_bboxes: list[np.ndarray],
    detection_bboxes: list[np.ndarray],
    iou_threshold: float = 0.3,
    use_mahalanobis_gate: bool = True,
) -> AssociationResult:
    """Hungarian-match predicted track boxes against detection boxes.

    ``predicted_bboxes[i]`` must correspond to ``tracks[i]`` (i.e. already
    advanced through :func:`semantic_mapping.tracking.predict`) so the
    Mahalanobis gate is evaluated against the correct prior covariance.
    """
    result = AssociationResult()
    n_tracks, n_dets = len(tracks), len(detection_bboxes)

    if n_tracks == 0 or n_dets == 0:
        result.unmatched_tracks = list(range(n_tracks))
        result.unmatched_detections = list(range(n_dets))
        return result

    cost = _cost_matrix(predicted_bboxes, detection_bboxes, iou_threshold)
    track_idx, det_idx = linear_sum_assignment(cost)

    matched_tracks, matched_dets = set(), set()
    for i, j in zip(track_idx, det_idx):
        if cost[i, j] >= INVALID_COST:
            continue
        if use_mahalanobis_gate and not mahalanobis_gate(tracks[i], detection_bboxes[j]):
            continue
        result.matches.append((int(i), int(j)))
        matched_tracks.add(i)
        matched_dets.add(j)

    result.unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
    result.unmatched_detections = [j for j in range(n_dets) if j not in matched_dets]
    return result


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
