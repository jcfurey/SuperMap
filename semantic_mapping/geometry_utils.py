"""Geometric primitives shared by the tracking, association, and mapping modules.

Implements the projection / back-projection operators and box overlap
predicates used throughout Sec. IV of the SuperMap paper, e.g. the
motion-compensated projection in Eq. (5) and the geometric ``On`` predicate
in Sec. IV-C.
"""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> Array:
    """Convert a unit quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)],
    ], dtype=np.float64)


def rotation_matrix_to_quaternion(R: Array) -> Array:
    """Convert a 3x3 rotation matrix to a unit quaternion (x, y, z, w)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


def se3_from_translation_quaternion(translation: Array, quaternion_xyzw: Array) -> Array:
    """Build a 4x4 SE(3) transform from a translation and an (x, y, z, w) quaternion."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_to_rotation_matrix(*quaternion_xyzw)
    T[:3, 3] = translation
    return T


def invert_se3(T: Array) -> Array:
    """Invert a 4x4 SE(3) homogeneous transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=T.dtype)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def compose_se3(T_a: Array, T_b: Array) -> Array:
    """Compose two 4x4 SE(3) transforms: T_a * T_b."""
    return T_a @ T_b


def transform_points(T: Array, points: Array) -> Array:
    """Apply a 4x4 SE(3) transform to an (N, 3) array of points."""
    points = np.atleast_2d(points)
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    out = (T @ homogeneous.T).T
    return out[:, :3]


def project_points(K: Array, T_world_to_cam: Array, points_world: Array) -> tuple[Array, Array]:
    """Project 3D world points into the image plane.

    Implements Eq. (5): c_hat = pi(K * P_t^-1 * X_i), where ``T_world_to_cam``
    is P_t^-1 (the world-to-camera transform) so that callers pass the
    current camera pose P_t directly via :func:`invert_se3`.

    Returns
    -------
    pixels : (N, 2) array of (u, v) image coordinates.
    depths : (N,) array of camera-frame depths (z). Points with non-positive
        depth are behind the camera and should be discarded by the caller.
    """
    points_cam = transform_points(T_world_to_cam, points_world)
    depths = points_cam[:, 2]
    safe_depths = np.where(np.abs(depths) < 1e-9, 1e-9, depths)
    homogeneous_pixels = (K @ points_cam.T).T
    pixels = homogeneous_pixels[:, :2] / safe_depths[:, None]
    return pixels, depths


def project_point(K: Array, T_world_to_cam: Array, point_world: Array) -> tuple[Array, float]:
    """Single-point convenience wrapper around :func:`project_points`."""
    pixels, depths = project_points(K, T_world_to_cam, point_world.reshape(1, 3))
    return pixels[0], float(depths[0])


def project_bbox3d(K: Array, T_world_to_cam: Array, bbox3d: Array) -> tuple[Array, float]:
    """Project a world-frame axis-aligned box into the image.

    Returns the [x1, y1, x2, y2] envelope of the eight projected corners and
    the smallest corner depth; a non-positive depth means part of the box is
    behind the camera and the envelope is not meaningful.
    """
    xmin, ymin, zmin, xmax, ymax, zmax = bbox3d
    corners = np.array([[x, y, z] for x in (xmin, xmax) for y in (ymin, ymax) for z in (zmin, zmax)])
    pixels, depths = project_points(K, T_world_to_cam, corners)
    box = np.array([pixels[:, 0].min(), pixels[:, 1].min(), pixels[:, 0].max(), pixels[:, 1].max()])
    return box, float(depths.min())


def clip_bbox_to_image(bbox: Array, width: int, height: int) -> Array | None:
    """Intersect an [x1, y1, x2, y2] box with the image; None when nothing is left."""
    x1, y1 = max(float(bbox[0]), 0.0), max(float(bbox[1]), 0.0)
    x2, y2 = min(float(bbox[2]), float(width)), min(float(bbox[3]), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def back_project_depth(K: Array, depth: Array, mask: Array | None = None) -> Array:
    """Back-project a depth image (or masked subset) into the camera frame.

    Parameters
    ----------
    K : (3, 3) camera intrinsic matrix.
    depth : (H, W) depth image in meters, 0/NaN entries are treated as invalid.
    mask : optional (H, W) boolean array restricting which pixels to unproject.

    Returns
    -------
    (N, 3) array of camera-frame 3D points, one per valid pixel.
    """
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= mask.astype(bool)
    vs, us = np.nonzero(valid)
    if us.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    z = depth[vs, us].astype(np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (us.astype(np.float64) - cx) * z / fx
    y = (vs.astype(np.float64) - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def rasterize_depth(points_cam: Array, K: Array, width: int, height: int) -> Array:
    """Z-buffer rasterization of camera-frame points into a dense depth image.

    Used in live mode to turn the synchronized LiDAR point cloud into the
    per-pixel raw sensor depth D(u) needed by the geometric-consistency
    update (Eq. 7-9).
    """
    depth = np.zeros((height, width), dtype=np.float64)
    if points_cam.shape[0] == 0:
        return depth

    z = points_cam[:, 2]
    in_front = z > 1e-6
    points_cam = points_cam[in_front]
    z = z[in_front]
    if points_cam.shape[0] == 0:
        return depth

    homogeneous_pixels = (K @ points_cam.T).T
    pixels = homogeneous_pixels[:, :2] / z[:, None]
    us = np.round(pixels[:, 0]).astype(np.int64)
    vs = np.round(pixels[:, 1]).astype(np.int64)
    in_frame = (us >= 0) & (us < width) & (vs >= 0) & (vs < height)
    us, vs, z = us[in_frame], vs[in_frame], z[in_frame]

    # Sort far-to-near so the nearest point wins the z-buffer write for each pixel.
    order = np.argsort(-z)
    depth[vs[order], us[order]] = z[order]
    return depth


def depth_consistency_mask(depths: Array, mad_factor: float = 3.0, min_tolerance: float = 0.05) -> Array:
    """Reject background clutter within a loose detection (box-only, no mask):
    keep only points whose depth lies within ``mad_factor`` median-absolute-
    deviations of the region's median depth. A detector's box commonly
    includes some background around the object's true silhouette (especially
    without SAM-style mask refinement); back-projecting the whole box would
    otherwise pull the fused 3D point set -- and hence the object's centroid
    and bbox3d -- toward whatever surface is behind it.
    """
    if depths.size == 0:
        return np.zeros(0, dtype=bool)
    median = np.median(depths)
    mad = np.median(np.abs(depths - median))
    tolerance = max(mad_factor * mad, min_tolerance)
    return np.abs(depths - median) <= tolerance


def iou_xyxy(box_a: Array, box_b: Array) -> float:
    """Intersection-over-union of two axis-aligned 2D boxes in (x1, y1, x2, y2) form."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area
    if union <= 1e-12:
        return 0.0
    return float(inter_area / union)


def bbox3d_from_points(points: Array) -> Array:
    """Axis-aligned 3D bounding box [xmin, ymin, zmin, xmax, ymax, zmax] for a point set."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return np.concatenate([mins, maxs])


def iou_xy(bbox3d_a: Array, bbox3d_b: Array) -> float:
    """IoU of the XY footprints of two axis-aligned 3D boxes (Sec. IV-C ``On`` predicate)."""
    box_a_xy = np.array([bbox3d_a[0], bbox3d_a[1], bbox3d_a[3], bbox3d_a[4]])
    box_b_xy = np.array([bbox3d_b[0], bbox3d_b[1], bbox3d_b[3], bbox3d_b[4]])
    return iou_xyxy(box_a_xy, box_b_xy)


def iou_3d(bbox3d_a: Array, bbox3d_b: Array) -> float:
    """IoU of two axis-aligned 3D boxes [xmin, ymin, zmin, xmax, ymax, zmax]."""
    inter_min = np.maximum(bbox3d_a[:3], bbox3d_b[:3])
    inter_max = np.minimum(bbox3d_a[3:], bbox3d_b[3:])
    inter_dims = np.clip(inter_max - inter_min, 0.0, None)
    inter_vol = float(np.prod(inter_dims))
    vol_a = float(np.prod(np.clip(bbox3d_a[3:] - bbox3d_a[:3], 0.0, None)))
    vol_b = float(np.prod(np.clip(bbox3d_b[3:] - bbox3d_b[:3], 0.0, None)))
    union = vol_a + vol_b - inter_vol
    if union <= 1e-12:
        return 0.0
    return inter_vol / union


def centroid(bbox3d: Array) -> Array:
    """Center of an axis-aligned 3D box [xmin, ymin, zmin, xmax, ymax, zmax]."""
    return np.array([
        (bbox3d[0] + bbox3d[3]) / 2.0,
        (bbox3d[1] + bbox3d[4]) / 2.0,
        (bbox3d[2] + bbox3d[5]) / 2.0,
    ])
