"""Shared data types for the SuperMap instance-layer and topological-layer pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ObjectStatus(str, Enum):
    """Lifecycle status of a mapped object instance, reported in the JSON schema."""

    ACTIVE = "active"
    """Currently observable and geometrically consistent."""

    OCCLUDED = "occluded"
    """Not currently observed but not yet confirmed removed (Sec. IV-B.1 re-activation)."""

    DISAPPEARED = "disappeared"
    """Confirmed removed: geometric evidence contradicts continued existence (Eq. 9)."""

    TENTATIVE = "tentative"
    """Newly spawned track that has not yet accumulated enough evidence to confirm."""


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics and image size."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass
class Detection2D:
    """A single 2D open-vocabulary detection for one frame."""

    bbox: np.ndarray
    """Image-space box [x1, y1, x2, y2] in pixels."""

    label: str
    score: float
    mask: np.ndarray | None = None
    """Optional (H, W) boolean instance mask, e.g. from SAM2."""

    embedding: np.ndarray | None = None
    """Optional open-vocabulary embedding (e.g. CLIP) for label-free matching."""

    def validate_mask(self, image_shape: tuple[int, int]) -> None:
        """Masks must address pixels in the original image, not a resized model input."""
        if self.mask is not None:
            if self.mask.shape != image_shape or self.mask.dtype != np.bool_:
                raise ValueError(
                    f"detection {self.label!r} needs a boolean mask of shape {image_shape}; "
                    f"got {self.mask.shape} with dtype {self.mask.dtype}")


@dataclass
class StampedPose:
    """A timestamped SE(3) world-from-body (or world-from-camera) pose."""

    stamp: float
    T_world_from_frame: np.ndarray
    """4x4 homogeneous transform."""


@dataclass
class Observation:
    """Synchronized per-frame input to the pipeline: Q_t = {C_t, D_t} plus pose P_t."""

    stamp: float
    pose: StampedPose
    intrinsics: CameraIntrinsics
    frame_id: int | None = None
    """Monotonic source-frame index, used by pre-baked detector backends."""

    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None
    """Camera-frame depth image (H, W) in meters, aligned to ``intrinsics``. In
    live mode this is produced by rasterizing the synchronized LiDAR/point
    cloud into the camera frame; see ``node.py``."""

    detections: list[Detection2D] = field(default_factory=list)


@dataclass
class TrackKalmanState:
    """Hybrid tracklet state S_i(t) = [c_i, s_i, cdot_i] from Eq. (4)."""

    state: np.ndarray
    """6-vector [x, y, w, h, vx, vy] (image centroid, box size, image-plane velocity)."""

    covariance: np.ndarray
    """6x6 state covariance."""


@dataclass
class ObjectInstance:
    """An entry O_j_t of the global map M_t."""

    instance_id: int
    label_belief: dict[str, float]
    """Categorical belief P(L_j = c) over open-vocabulary labels, Eq. (10)."""

    points_world: np.ndarray
    """Fused world-frame point set backing the voxelized object volume, (N, 3)."""

    point_log_odds: np.ndarray
    """Per-point accumulated log-odds L(o_k | Q_1:t) from Eq. (8), shape (N,)."""

    bbox3d: np.ndarray
    """Axis-aligned world-frame box [xmin, ymin, zmin, xmax, ymax, zmax]."""

    status: ObjectStatus
    track: TrackKalmanState
    first_seen_stamp: float
    latest_stamp: float
    frames_since_seen: int = 0
    hits: int = 0
    points_contradicted: int = 0
    """Points pruned because geometric evidence contradicted them since the
    instance was last detected. Pruning removes the contradicted points, so
    without this count an object whose bulk has been ruled out but whose
    floor-contact points still coincide with the floor would look intact."""

    point_membership: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Per-point log-odds that the point belongs to *this* instance (Sec. IV-B.4,
    "remove object points whose posterior belief is too small"), shape (N,).
    Geometric consistency (``point_log_odds``) only says a surface exists at the
    point; this says whether that surface is part of the object the detector
    keeps reporting -- it goes negative when the point is visible but keeps
    projecting outside the detected region, which is how background caught in
    a loose box gets pruned instead of inflating the object's 3D extent."""

    trajectory: list[tuple[float, np.ndarray, str]] = field(default_factory=list)
    """Temporal edge history E_t: (stamp, world centroid, status) samples tracing this
    instance's trajectory (Sec. IV-C), used for temporal-edge queries such as
    "where was this object before it disappeared"."""

    embedding: np.ndarray | None = None
    """Unit-norm running mean of the appearance descriptors of the detections
    fused into this instance (semantic_mapping.appearance); what lets the
    same physical object keep its ID after it disappears and turns up elsewhere."""

    embedding_count: int = 0

    @property
    def label(self) -> str:
        if not self.label_belief:
            return "unknown"
        return max(self.label_belief.items(), key=lambda kv: kv[1])[0]

    @property
    def label_confidence(self) -> float:
        if not self.label_belief:
            return 0.0
        return max(self.label_belief.values())

    @property
    def center(self) -> np.ndarray:
        from semantic_mapping.geometry_utils import centroid

        return centroid(self.bbox3d)

    @property
    def mean_log_odds(self) -> float:
        if self.point_log_odds.size == 0:
            return 0.0
        return float(np.mean(self.point_log_odds))
