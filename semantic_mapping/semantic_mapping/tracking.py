"""Hybrid 3D-to-2D motion-compensated tracker (Sec. IV-B.1, IV-B.2).

Standard 2D trackers such as ByteTrack assume roughly linear image-plane
motion and struggle under rapid ego-motion. SuperMap instead predicts each
track's image-space centroid by re-projecting its current 3D world centroid
through the *current* camera pose (Eq. 5), and only uses a constant-velocity
Kalman filter to smooth box size and residual image-plane motion (Eq. 6).
"""
from __future__ import annotations

import numpy as np

from semantic_mapping.geometry_utils import centroid, clip_bbox_to_image, invert_se3, project_bbox3d, project_point
from semantic_mapping.types import TrackKalmanState

# State layout: [cx, cy, w, h, vx, vy]
STATE_DIM = 6
MEAS_DIM = 4

H = np.zeros((MEAS_DIM, STATE_DIM), dtype=np.float64)
H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0


def transition_matrix(dt: float) -> np.ndarray:
    """Constant-velocity transition matrix F for a given time step."""
    F = np.eye(STATE_DIM, dtype=np.float64)
    F[0, 4] = dt
    F[1, 5] = dt
    return F


def process_noise(dt: float, position_noise: float = 640.0, size_noise: float = 250.0,
                   velocity_noise: float = 200.0) -> np.ndarray:
    """Diagonal process-noise covariance Q (px^2 per second); larger dt admits more drift.

    The projection-based prior in :func:`predict` replaces the centroid
    prediction outright rather than propagating it through a calibrated
    motion model, so the residual uncertainty it carries -- from pose drift,
    3D centroid estimation error, and viewpoint change -- is deliberately
    generous (tens of pixels of std at a typical 10 Hz step) rather than the
    sub-pixel drift a textbook constant-velocity filter would assume; too
    tight a Q here makes the covariance overconfident and the association
    gate below reject perfectly good matches.
    """
    q = np.array([
        position_noise * dt,
        position_noise * dt,
        size_noise * dt,
        size_noise * dt,
        velocity_noise * dt,
        velocity_noise * dt,
    ], dtype=np.float64)
    return np.diag(q)


def measurement_noise(position_noise: float = 25.0, size_noise: float = 25.0) -> np.ndarray:
    return np.diag([position_noise, position_noise, size_noise, size_noise]).astype(np.float64)


def bbox_to_measurement(bbox: np.ndarray) -> np.ndarray:
    """Convert an [x1, y1, x2, y2] box into a [cx, cy, w, h] measurement."""
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dtype=np.float64)


def measurement_to_bbox(measurement: np.ndarray) -> np.ndarray:
    cx, cy, w, h = measurement
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float64)


def init_track(bbox: np.ndarray, initial_velocity_var: float = 1e3) -> TrackKalmanState:
    """Create a new tracklet state S_i(0) from its first detection."""
    cx, cy, w, h = bbox_to_measurement(bbox)
    state = np.array([cx, cy, w, h, 0.0, 0.0], dtype=np.float64)
    covariance = np.diag([10.0, 10.0, 10.0, 10.0, initial_velocity_var, initial_velocity_var]).astype(np.float64)
    return TrackKalmanState(state=state, covariance=covariance)


def predict(
    track: TrackKalmanState,
    dt: float,
    *,
    K: np.ndarray | None = None,
    T_world_from_cam: np.ndarray | None = None,
    object_centroid_world: np.ndarray | None = None,
    object_bbox3d_world: np.ndarray | None = None,
    image_size: tuple[int, int] | None = None,
    projection_prior_weight: float = 1.0,
    size_prior_weight: float = 0.0,
) -> TrackKalmanState:
    """Predict the tracklet forward by ``dt``, implementing Eq. (5)-(6).

    When camera intrinsics/pose and the object's 3D state are available, the
    predicted image centroid is replaced (or blended, via
    ``projection_prior_weight`` in [0, 1]) by the re-projection
    c_hat_i(t) = pi(K * P_t^-1 * X_i), which remains valid under aggressive
    ego-motion where a linear image-plane velocity model would fail.

    Given the object's 3D box (``object_bbox3d_world``) and the image size,
    the prior is the centre of the projected box's *visible* part rather
    than the projected centroid: a detector boxes only what is inside the
    frame, so for an object cut by the image border the two differ by up to
    half the object's width, enough to hand its detection to a neighbour.
    ``size_prior_weight`` optionally blends the box size toward the visible
    projected extent as well. Falls back to the centroid projection when the
    box has corners behind the camera or lies entirely outside the image.
    """
    F = transition_matrix(dt)
    Q = process_noise(max(dt, 1e-3))

    predicted_state = F @ track.state
    predicted_covariance = F @ track.covariance @ F.T + Q

    if object_centroid_world is None and object_bbox3d_world is not None:
        object_centroid_world = centroid(object_bbox3d_world)

    if K is not None and T_world_from_cam is not None and object_centroid_world is not None:
        T_cam_from_world = invert_se3(T_world_from_cam)
        prior: np.ndarray | None = None  # [cx, cy, w, h] of the (visible) projected object
        if object_bbox3d_world is not None:
            box2d, min_depth = project_bbox3d(K, T_cam_from_world, object_bbox3d_world)
            if min_depth > 0:
                visible = clip_bbox_to_image(box2d, *image_size) if image_size is not None else box2d
                if visible is not None:
                    prior = bbox_to_measurement(visible)
        if prior is None:
            pixel, depth = project_point(K, T_cam_from_world, object_centroid_world)
            if depth > 0:
                prior = np.array([pixel[0], pixel[1], predicted_state[2], predicted_state[3]])
        if prior is not None:
            w = float(np.clip(projection_prior_weight, 0.0, 1.0))
            s = float(np.clip(size_prior_weight, 0.0, 1.0))
            predicted_state[0:2] = (1 - w) * predicted_state[0:2] + w * prior[0:2]
            predicted_state[2:4] = (1 - s) * predicted_state[2:4] + s * prior[2:4]

    return TrackKalmanState(state=predicted_state, covariance=predicted_covariance)


def update(track: TrackKalmanState, measurement_bbox: np.ndarray) -> TrackKalmanState:
    """Standard Kalman measurement update given an associated detection box."""
    R = measurement_noise()
    z = bbox_to_measurement(measurement_bbox)

    innovation = z - H @ track.state
    innovation_covariance = H @ track.covariance @ H.T + R
    kalman_gain = track.covariance @ H.T @ np.linalg.inv(innovation_covariance)

    new_state = track.state + kalman_gain @ innovation
    identity = np.eye(STATE_DIM)
    new_covariance = (identity - kalman_gain @ H) @ track.covariance

    return TrackKalmanState(state=new_state, covariance=new_covariance)


def current_bbox(track: TrackKalmanState) -> np.ndarray:
    return measurement_to_bbox(track.state[:4])


def current_velocity(track: TrackKalmanState) -> np.ndarray:
    return track.state[4:6]


def mahalanobis_gate(track: TrackKalmanState, measurement_bbox: np.ndarray, gate_threshold: float = 9.4877) -> bool:
    """Chi-square gate (4 DoF, ~95%) on the innovation to reject implausible matches."""
    R = measurement_noise()
    z = bbox_to_measurement(measurement_bbox)
    innovation = z - H @ track.state
    innovation_covariance = H @ track.covariance @ H.T + R
    distance = float(innovation.T @ np.linalg.inv(innovation_covariance) @ innovation)
    return distance <= gate_threshold
