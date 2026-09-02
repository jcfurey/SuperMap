"""Probabilistic geometric consistency update (Sec. IV-B.3, Eq. 7-9).

Each fused 3D point belonging to an object is treated as a random variable
whose occupancy is confirmed or contradicted by re-projecting it into the
current depth frame and comparing the projected depth against the raw
sensor reading. The resulting per-point evidence is fused recursively via a
log-odds occupancy filter (Eq. 8), which is what lets SuperMap prune stale
map content (occlusions, relocations, removals) without discarding points
that are merely temporarily out of view.
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np

from semantic_mapping.geometry_utils import invert_se3, project_points

# Clamp bounds for the log-odds accumulator, standard practice in occupancy-grid
# mapping to keep a single outlier measurement from permanently pinning a value.
LOG_ODDS_MIN = -8.0
LOG_ODDS_MAX = 8.0


class GeometricState(IntEnum):
    """Classification codes, stored in a plain int array (not an object array of
    enum members): vectorized ``==`` against a ``str``-mixin Enum scalar is
    unreliable across numpy versions, silently comparing false even when the
    array does hold the expected member. Plain ints sidestep that entirely.
    """

    OBSERVABLE = 0
    """|delta_d| <= tau_eps: sensor confirms the point (Eq. 9, case 1)."""

    UNOBSERVABLE = 1
    """delta_d > tau_eps: point lies behind the observed surface, i.e. occluded."""

    DISAPPEARED = 2
    """delta_d < -tau_eps: point lies in front of the observed surface, i.e. empty space now."""

    OUT_OF_VIEW = 3
    """Projects outside the image or has no valid depth reading; no evidence either way."""


def logit(p: float | np.ndarray) -> float | np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def log_odds_to_prob(log_odds: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-log_odds))


def gaussian_likelihood(delta_d: np.ndarray, sigma: float) -> np.ndarray:
    """p(delta_d) ~ N(0, sigma^2), the Gaussian sensor-noise model motivating tau_eps."""
    sigma = max(sigma, 1e-6)
    return np.exp(-0.5 * (delta_d / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def project_and_classify(
    K: np.ndarray,
    T_world_from_cam: np.ndarray,
    depth_image: np.ndarray,
    points_world: np.ndarray,
    tau_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify each world-frame point per Eq. (9) and return where it projects.

    Returns
    -------
    states : int array of :class:`GeometricState` codes, one per point.
    delta_d : array of signed depth residuals (NaN where undefined).
    pixels : (N, 2) int array of rounded (u, v) image coordinates; only
        meaningful for points whose state is not ``OUT_OF_VIEW``.
    """
    n = points_world.shape[0]
    states = np.full(n, int(GeometricState.OUT_OF_VIEW), dtype=np.int64)
    delta_d = np.full(n, np.nan, dtype=np.float64)
    pixels_int = np.full((n, 2), -1, dtype=np.int64)

    if n == 0:
        return states, delta_d, pixels_int

    T_cam_from_world = invert_se3(T_world_from_cam)
    pixels, d_proj = project_points(K, T_cam_from_world, points_world)

    h, w = depth_image.shape
    us = np.round(pixels[:, 0]).astype(np.int64)
    vs = np.round(pixels[:, 1]).astype(np.int64)
    in_front = d_proj > 1e-6
    in_frame = (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
    valid = in_front & in_frame
    pixels_int[valid, 0] = us[valid]
    pixels_int[valid, 1] = vs[valid]

    sensor_depth = np.full(n, np.nan, dtype=np.float64)
    sensor_depth[valid] = depth_image[vs[valid], us[valid]]
    has_reading = valid & np.isfinite(sensor_depth) & (sensor_depth > 0)

    delta = d_proj - sensor_depth
    delta_d[has_reading] = delta[has_reading]

    observable = has_reading & (np.abs(delta) <= tau_eps)
    unobservable = has_reading & (delta > tau_eps)
    disappeared = has_reading & (delta < -tau_eps)

    states[observable] = GeometricState.OBSERVABLE
    states[unobservable] = GeometricState.UNOBSERVABLE
    states[disappeared] = GeometricState.DISAPPEARED
    # remaining entries keep GeometricState.OUT_OF_VIEW

    return states, delta_d, pixels_int


def classify_points(
    K: np.ndarray,
    T_world_from_cam: np.ndarray,
    depth_image: np.ndarray,
    points_world: np.ndarray,
    tau_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify each world-frame point per Eq. (9); see :func:`project_and_classify`."""
    states, delta_d, _pixels = project_and_classify(K, T_world_from_cam, depth_image, points_world, tau_eps)
    return states, delta_d


def inverse_sensor_model(
    states: np.ndarray,
    p_hit: float = 0.9,
    p_miss: float = 0.1,
    p_unknown: float = 0.5,
) -> np.ndarray:
    """Map the thresholded geometric-consistency classification to P(o_k | Q_t)."""
    probabilities = np.full(states.shape, p_unknown, dtype=np.float64)
    probabilities[states == GeometricState.OBSERVABLE] = p_hit
    probabilities[states == GeometricState.DISAPPEARED] = p_miss
    # UNOBSERVABLE (occluded) and OUT_OF_VIEW carry no evidence -> p_unknown,
    # which is a logit of 0 and therefore leaves the log-odds unchanged.
    return probabilities


def update_log_odds(prior_log_odds: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Recursive log-odds fusion, Eq. (8): L(o_k | Q_1:t) = L(o_k | Q_1:t-1) + logit(P(o_k | Q_t))."""
    updated = prior_log_odds + logit(probabilities)
    return np.clip(updated, LOG_ODDS_MIN, LOG_ODDS_MAX)


def update_object_points(
    K: np.ndarray,
    T_world_from_cam: np.ndarray,
    depth_image: np.ndarray,
    points_world: np.ndarray,
    prior_log_odds: np.ndarray,
    tau_eps: float,
    p_hit: float = 0.9,
    p_miss: float = 0.1,
    p_unknown: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One fused geometric-consistency step for all points of an object.

    Returns the updated per-point log-odds, the raw classification (so callers
    can restrict semantic updates to points classified
    :attr:`GeometricState.OBSERVABLE`), and each point's projected pixel.
    """
    states, _delta_d, pixels = project_and_classify(K, T_world_from_cam, depth_image, points_world, tau_eps)
    probabilities = inverse_sensor_model(states, p_hit=p_hit, p_miss=p_miss, p_unknown=p_unknown)
    new_log_odds = update_log_odds(prior_log_odds, probabilities)
    return new_log_odds, states, pixels


def prune_mask(log_odds: np.ndarray, prune_threshold: float = -1.5) -> np.ndarray:
    """Boolean mask of points whose occupancy evidence has fallen below threshold."""
    return log_odds < prune_threshold


def occupied_fraction(log_odds: np.ndarray) -> float:
    """Fraction of points not contradicted by evidence.

    A point at the neutral prior (log-odds 0, p = 0.5) counts as occupied: it
    was just back-projected from a detection and simply hasn't been re-observed
    yet, which is not evidence of absence. Only accumulated negative evidence
    (Eq. 9 "disappeared" classifications) moves a point below the line.
    """
    if log_odds.size == 0:
        return 0.0
    return float(np.mean(log_odds >= 0.0))
