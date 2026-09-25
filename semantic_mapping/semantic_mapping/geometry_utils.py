"""Geometric primitives shared by the tracking, association, and mapping modules.

Implements the projection / back-projection operators and box overlap
predicates used throughout Sec. IV of the SuperMap paper, e.g. the
motion-compensated projection in Eq. (5) and the geometric ``On`` predicate
in Sec. IV-C.
"""
from __future__ import annotations

import numpy as np

from semantic_mapping import native

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


VOXEL_KEY_BITS = 21
"""Voxel coordinates are packed three-per-int64 with 21 bits each, i.e. any
map within +-2^20 voxels of the origin (+-52 km at 5 cm) takes the fast path."""


def pack_voxel_keys(cells: Array) -> Array | None:
    """One int64 per integer ``(x, y, z)`` cell, ordered like the rows themselves
    (lexicographically), or None when a coordinate lies outside +-2^20.

    Sorting, de-duplicating and searching the keys is a 1-D operation on
    integers; row-wise ``np.unique(axis=0)`` and structured-view searches over
    the same cells are several times slower (doc/audit-2026-09.md, P2)."""
    offset = 1 << (VOXEL_KEY_BITS - 1)
    shifted = np.asarray(cells, dtype=np.int64).reshape(-1, 3) + offset
    if shifted.size and (shifted.min() < 0 or shifted.max() >= 1 << VOXEL_KEY_BITS):
        return None
    return (shifted[:, 0] << (2 * VOXEL_KEY_BITS)) | (shifted[:, 1] << VOXEL_KEY_BITS) | shifted[:, 2]


def mask_bounds(mask: Array) -> tuple[int, int, int, int] | None:
    """Rows ``[y1, y2)`` and columns ``[x1, x2)`` holding a boolean mask's pixels,
    or None for an empty mask. Two row/column reductions, far cheaper than
    listing every pixel of a full-image mask."""
    rows = np.flatnonzero(mask.any(axis=1))
    if not rows.size:
        return None
    y1, y2 = int(rows[0]), int(rows[-1]) + 1
    cols = np.flatnonzero(mask[y1:y2].any(axis=0))
    return y1, y2, int(cols[0]), int(cols[-1]) + 1


def back_project_depth(
    K: Array, depth: Array, mask: Array | None = None, *,
    max_points: int | None = None, depth_mad_factor: float = 0.0, depth_min_tolerance: float = 0.05,
    pixel_origin: tuple[int, int] = (0, 0),
) -> Array:
    """Back-project a depth image (or masked subset) into the camera frame.

    Parameters
    ----------
    K : (3, 3) camera intrinsic matrix.
    depth : (H, W) depth image in meters, 0/NaN entries are treated as invalid.
    mask : optional (H, W) boolean array restricting which pixels to unproject.
    max_points : optional deterministic sample limit, applied before projection.
    depth_mad_factor : optional median-absolute-deviation gate; 0 disables it.
    depth_min_tolerance : minimum depth tolerance in meters when the gate is enabled.
    pixel_origin : (column, row) of ``depth[0, 0]`` in the image ``K`` describes,
        so a crop back-projects exactly as the same pixels of the full image.

    Returns
    -------
    (N, 3) array of camera-frame 3D points after optional filtering/sampling.
    """
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= mask.astype(bool)
    vs, us = np.nonzero(valid)
    if us.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    z = depth[vs, us].astype(np.float64)
    us, vs = us + pixel_origin[0], vs + pixel_origin[1]
    if depth_mad_factor > 0:
        keep = depth_consistency_mask(z, mad_factor=depth_mad_factor, min_tolerance=depth_min_tolerance)
        vs, us, z = vs[keep], us[keep], z[keep]
    if max_points is not None and us.size > max_points:
        # Same deterministic sample as subsampling the projected cloud, but
        # avoid computing and allocating XYZ for pixels that will be discarded.
        indices = np.random.default_rng(0).choice(us.size, size=max_points, replace=False)
        vs, us, z = vs[indices], us[indices], z[indices]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (us.astype(np.float64) - cx) * z / fx
    y = (vs.astype(np.float64) - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def splat_depth_buffer(us: Array, vs: Array, z: Array, focal: float, width: int, height: int,
                       radius_m: float, max_px: int) -> Array:
    """Flat per-pixel nearest depth where each point covers a square footprint.

    A point at depth ``z`` covers the pixels within ``round(focal * radius_m / z)``
    (at most ``max_px``) of its own, i.e. a disc of ``radius_m`` metres on its
    surface. ``us``/``vs`` are integer pixel indices inside the image. Unset
    pixels hold +inf; ``radius_m <= 0`` returns all +inf.
    """
    if native.kernels is not None:  # the same buffer, computed in one pass per point
        return native.kernels.splat_depth_buffer(us, vs, z, focal, width, height, radius_m, max_px)
    buffer = np.full(width * height, np.inf)
    if radius_m <= 0 or not len(z):
        return buffer
    radius = np.minimum(np.floor(focal * radius_m / z + 0.5), max_px).astype(np.int64)
    for r in range(int(radius.max()) + 1):
        # Points whose footprint reaches Chebyshev ring r write that ring.
        sel = np.flatnonzero(radius >= r)
        if not len(sel):
            break
        offsets = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1) if max(abs(dx), abs(dy)) == r]
        for dx, dy in offsets:
            uu, vv = us[sel] + dx, vs[sel] + dy
            inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            np.minimum.at(buffer, vv[inside] * width + uu[inside], z[sel][inside])
    return buffer


def occlusion_visible(us: Array, vs: Array, z: Array, focal: float, width: int, height: int, *,
                      radius_m: float, max_px: int = 8, gap_m: float = 0.3, grid_px: int = 1,
                      keep_dense_surfaces: bool = False) -> Array:
    """Which projected points no nearer point's footprint hides from the camera.

    A sparse cloud leaves holes between the returns of a foreground surface,
    and a one-pixel z-buffer lets the background behind it show through them
    (from a LiDAR mounted apart from the camera, also returns the camera
    cannot see at all). A point is hidden when the footprint of another point
    (:func:`splat_depth_buffer`) covers its pixel more than ``gap_m`` in front
    of it. ``grid_px > 1`` decides this on cells of that many pixels, which
    bounds the cost for high-resolution cameras; a LiDAR's angular spacing is
    coarser than such cells anyway. ``radius_m <= 0`` keeps every point.

    ``keep_dense_surfaces`` keeps a point whose own surface is sampled densely
    around it: a reading within ``gap_m`` of its depth on an adjacent pixel
    both vertically and horizontally. There the one-pixel z-buffer is already
    exact, and footprints would only hide genuine background beside a nearer
    silhouette (a dense depth-camera cloud lost about a tenth of its readings).
    A spinning LiDAR's rings lie several pixels apart, so its returns never
    qualify. Accumulated clouds can: do not use it for map snapshots.
    """
    visible = np.ones(len(z), dtype=bool)
    if radius_m <= 0 or not len(z):
        return visible
    if native.kernels is not None:  # the same test without full-image temporaries
        return native.kernels.occlusion_visible(us, vs, z, focal, width, height, radius_m, max_px, gap_m, int(grid_px),
                                                keep_dense_surfaces)
    dense = np.zeros(len(z), dtype=bool)
    if keep_dense_surfaces:
        nearest = np.full(width * height, np.inf)
        np.minimum.at(nearest, vs * width + us, z)

        def neighbour(du: int, dv: int) -> Array:
            uu, vv = us + du, vs + dv
            inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            found = np.zeros(len(z), dtype=bool)
            found[inside] = np.abs(nearest[vv[inside] * width + uu[inside]] - z[inside]) <= gap_m
            return found

        dense = (neighbour(-1, 0) | neighbour(1, 0)) & (neighbour(0, -1) | neighbour(0, 1))
    grid = max(int(grid_px), 1)
    if grid > 1:
        us, vs = us // grid, vs // grid
        width, height = -(-width // grid), -(-height // grid)
        focal, max_px = focal / grid, max(-(-int(max_px) // grid), 1)
    buffer = splat_depth_buffer(us, vs, z, focal, width, height, radius_m, max_px)
    return dense | (z <= buffer[vs * width + us] + gap_m)


def occlusion_grid_for(width: int, height: int, cells_long_side: int = 640) -> int:
    """Occlusion grid (px) for :func:`occlusion_visible` that keeps about
    ``cells_long_side`` cells along the longer image side: exact up to VGA,
    and 15-22 ms added per frame from VGA to 5 MP. The cells stay finer than a
    LiDAR's angular spacing; coarser cells only hide more of a grazing
    surface's farther returns (e.g. ground rings), never a nearer surface."""
    return max(1, int(np.ceil(max(width, height) / cells_long_side)))


def rasterize_depth(points_cam: Array, K: Array, width: int, height: int, *, splat_radius_m: float = 0.0,
                    splat_max_px: int = 8, occlusion_gap_m: float = 0.3, occlusion_grid_px: int = 1) -> Array:
    """Z-buffer rasterization of camera-frame points into a dense depth image.

    Used in live mode to turn the synchronized LiDAR point cloud into the
    per-pixel raw sensor depth D(u) needed by the geometric-consistency
    update (Eq. 7-9). With ``splat_radius_m > 0``, points hidden behind a
    nearer point's footprint are dropped first (:func:`occlusion_visible`,
    keeping densely sampled surfaces); the image still holds only real
    readings, one per pixel.
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
    if splat_radius_m > 0:
        visible = occlusion_visible(us, vs, z, max(K[0, 0], K[1, 1]), width, height, radius_m=splat_radius_m,
                                    max_px=splat_max_px, gap_m=occlusion_gap_m, grid_px=occlusion_grid_px,
                                    keep_dense_surfaces=True)
        us, vs, z = us[visible], vs[visible], z[visible]

    if native.kernels is not None:
        return native.kernels.depth_image(us, vs, z, width, height)
    # Per-pixel minimum through an unbuffered scatter: no sort, so a LiDAR
    # scan of a few hundred thousand points rasterizes in half the time of a
    # far-to-near z-buffer and the cost stays linear in the point count.
    flat = np.full(height * width, np.inf)
    np.minimum.at(flat, vs * width + us, z)
    depth[:] = np.where(np.isfinite(flat), flat, 0.0).reshape(height, width)
    return depth


def fill_sparse_depth(depth: Array, radius_px: int) -> Array:
    """Fill pixels without a depth reading from the nearest valid neighbours.

    A LiDAR scan rasterized into a camera covers a few percent of the pixels;
    every other pixel then carries no evidence for Eq. (7)-(9) and contributes
    no points when a detection is back-projected. Each empty pixel takes the
    *minimum* valid depth within ``radius_px``, and pixels with a reading are
    left untouched. Taking the minimum is the conservative choice for the
    consistency test: it can only make a filled pixel look closer, which turns
    a map point behind it into "occluded" (no evidence), never into free space
    (false disappearance). Pixels with no valid neighbour stay invalid (0).
    """
    depth = np.asarray(depth, dtype=np.float64)
    if radius_px <= 0:
        return depth
    if native.kernels is not None:  # one separable pass, no full-image temporaries
        return native.kernels.fill_sparse_depth(depth, int(radius_px))
    import cv2

    valid = np.isfinite(depth)
    valid &= depth > 0
    if valid.all() or not valid.any():
        return np.where(valid, depth, 0.0)
    # A square minimum filter is a grayscale erosion; OpenCV's is several times
    # faster than scipy.ndimage.minimum_filter and gives the same values
    # (replicated border = mode "nearest"). Few full-image passes: this runs on
    # every frame of a sparse depth source.
    size = 2 * int(radius_px) + 1
    filled = cv2.erode(np.where(valid, depth, np.inf), np.ones((size, size), np.uint8), borderType=cv2.BORDER_REPLICATE)
    np.copyto(filled, depth, where=valid)
    filled[np.isinf(filled)] = 0.0  # no valid neighbour
    return filled


def depth_consistency_mask(depths: Array, mad_factor: float = 3.0, min_tolerance: float = 0.05) -> Array:
    """Reject depth outliers within a detection region:
    keep only points whose depth lies within ``mad_factor`` median-absolute-
    deviations of the region's median depth. A detector's box commonly
    includes some background around the object's true silhouette (especially
    without SAM-style mask refinement); back-projecting the whole box would
    otherwise pull the fused 3D point set -- and hence the object's centroid
    and bbox3d -- toward whatever surface is behind it. Masked detections can
    opt into the same gate for mismatched depth pixels, with the tradeoff that
    a single median may exclude real parts of an object extending in depth.
    """
    if depths.size == 0:
        return np.zeros(0, dtype=bool)
    median = np.median(depths)
    mad = np.median(np.abs(depths - median))
    tolerance = max(mad_factor * mad, min_tolerance)
    return np.abs(depths - median) <= tolerance


def foreground_depth_mask(
    depths: Array, gap_m: float, min_points: int = 5, min_fraction: float = 0.1,
    *, pixels: Array | None = None, min_span: Array | None = None, select: str = "nearest",
) -> Array:
    """Keep the nearest supported depth layer, even when background dominates.

    Split sorted positive, finite sensor readings at depth gaps larger than
    ``gap_m``. Ignore isolated returns and select the first layer supported by
    both ``min_points`` and ``min_fraction`` of valid readings. No supported
    layer means unknown geometry. Call before filling sparse depth: invented
    neighbours must not count as independent support. This is intended for
    compact foreground objects, not structures extending through many depths.
    Optional image coordinates and minimum spans require each candidate to
    cover the silhouette in both image axes, rather than just its feet.
    ``select="largest"`` keeps the supported layer with the most readings
    instead (ties go to the nearer), for classes whose nearest layer is often
    the ground strip in front of them rather than the object.
    """
    if select not in ("nearest", "largest"):
        raise ValueError(f"unknown layer selection {select!r}")
    keep = np.zeros(depths.shape, dtype=bool)
    valid_indices = np.flatnonzero(np.isfinite(depths) & (depths > 0))
    if valid_indices.size == 0:
        return keep
    order = valid_indices[np.argsort(depths[valid_indices], kind="stable")]
    splits = np.flatnonzero(np.diff(depths[order]) > gap_m) + 1
    boundaries = np.concatenate(([0], splits, [order.size]))
    required = max(min_points, int(np.ceil(min_fraction * order.size)))
    supported = np.flatnonzero(np.diff(boundaries) >= required)
    chosen = None
    for i in supported:
        indices = order[boundaries[i]:boundaries[i + 1]]
        if min_span is not None:
            if pixels is None:
                raise ValueError("pixels are required with min_span")
            span = np.diff(np.percentile(pixels[indices], [5, 95], axis=0), axis=0)[0] + 1
            if np.any(span < min_span):
                continue
        if select == "nearest":
            chosen = indices
            break
        if chosen is None or indices.size > chosen.size:
            chosen = indices
    if chosen is not None:
        keep[chosen] = True
    return keep


GROUND_SURFACE_LABELS = ("floor", "ground", "road", "sidewalk", "pavement", "grass", "terrain", "carpet", "rug", "mat")
"""Classes that are the ground: their masks keep the returns on it."""


def fit_ground_plane(
    points: Array, tolerance_m: float = 0.15, max_slope: float = 0.25, cell_m: float = 0.5,
    iterations: int = 3, return_fitted: bool = False,
):
    """Local ground ``z = a x + b y + c`` under a set of world points (z up).

    Built for sparse LiDAR: the points are gridded in XY into ``cell_m``
    cells and only the lowest return of each cell is a ground candidate, so
    objects standing in a cell never pull the fit up as long as some ground
    return shares the cell. The fit is anchored at the 10th percentile of
    those minima (robust to a few below-ground noise returns); candidates are
    kept when a surface of at most ``max_slope`` through the anchor can reach
    them, a least-squares plane is fitted to those, and the fit is refined on
    candidates within ``tolerance_m`` of it. Too few or collinear candidates,
    or a fit steeper than ``max_slope``, fall back to the level anchor
    height. Returns ``[a, b, c]``, or None without points; with
    ``return_fitted``, ``(plane, fitted)`` where ``fitted`` is False for the
    level fallback (and None plane).
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if native.kernels is not None:
        # Same steps in one call (it runs once per detection); the plane agrees to rounding.
        result = native.kernels.fit_ground_plane(points, tolerance_m, max_slope, cell_m, iterations)
        if result is not None:
            return result if return_fitted else result[0]
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] == 0:
        return (None, False) if return_fitted else None
    # One int64 key per XY cell, ordered like the (x, y) cells themselves: a
    # row-wise np.unique here cost more than the rest of a detection's lifting.
    cells = np.floor(points[:, :2] / cell_m).astype(np.int64)
    cells -= cells.min(axis=0)
    cell = cells[:, 0] * (int(cells[:, 1].max()) + 1) + cells[:, 1]
    order = np.lexsort((points[:, 2], cell))
    first = np.ones(order.size, dtype=bool)
    first[1:] = cell[order][1:] != cell[order][:-1]
    seeds = points[order[first]]
    anchor = seeds[np.argsort(seeds[:, 2], kind="stable")[int(0.1 * (len(seeds) - 1))]]
    level = np.array([0.0, 0.0, anchor[2]])
    reach = tolerance_m + max_slope * np.linalg.norm(seeds[:, :2] - anchor[:2], axis=1)
    inliers = np.abs(seeds[:, 2] - anchor[2]) <= reach
    plane = level
    for _ in range(iterations):
        xy = seeds[inliers, :2]
        if xy.shape[0] < 3 or np.linalg.eigvalsh(np.cov(xy.T)).min() < (cell_m / 2) ** 2:
            return (level, False) if return_fitted else level
        A = np.column_stack((xy, np.ones(xy.shape[0])))
        plane = np.linalg.lstsq(A, seeds[inliers, 2], rcond=None)[0]
        if np.hypot(plane[0], plane[1]) > max_slope:
            return (level, False) if return_fitted else level
        inliers = np.abs(seeds[:, 2] - seeds[:, :2] @ plane[:2] - plane[2]) <= tolerance_m
    return (plane, True) if return_fitted else plane


def parse_size_limits(entries) -> dict[str, tuple[float, ...]]:
    """Parse ``"label:D"`` / ``"label:H,V"`` class size limits (metres).

    ``D`` bounds the 3D diagonal of an axis-aligned box; ``H,V`` bound the
    diagonal of its XY footprint (invariant to the object's heading) and its
    Z extent. The label may contain spaces; the last colon separates it.
    """
    limits: dict[str, tuple[float, ...]] = {}
    for entry in entries or ():
        label, sep, values = str(entry).rpartition(":")
        label = label.strip()
        try:
            numbers = tuple(float(v) for v in values.split(","))
        except ValueError:
            numbers = ()
        if not sep or not label or len(numbers) not in (1, 2) \
                or not all(np.isfinite(v) and v > 0 for v in numbers):
            raise ValueError(f"size limit {entry!r} must be 'label:diagonal' or 'label:horizontal,vertical'")
        if label in limits:
            raise ValueError(f"duplicate size limit for {label!r}")
        limits[label] = numbers
    return limits


def exceeds_size_limit(bbox3d: Array, limit: tuple[float, ...]) -> bool:
    """Whether an axis-aligned box violates a :func:`parse_size_limits` limit."""
    extent = np.clip(np.asarray(bbox3d[3:], dtype=np.float64) - bbox3d[:3], 0.0, None)
    if len(limit) == 1:
        return bool(np.linalg.norm(extent) > limit[0])
    return bool(np.hypot(extent[0], extent[1]) > limit[0] or extent[2] > limit[1])


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


def bbox3d_from_points(points: Array, trim_percentile: float = 0.0) -> Array:
    """Axis-aligned 3D bounding box [xmin, ymin, zmin, xmax, ymax, zmax] for a point set."""
    if not 0.0 <= trim_percentile < 50.0:
        raise ValueError("bbox trim percentile must be in [0, 50)")
    if trim_percentile > 0:
        # Optional robust bounds: sparse depth outliers must not drag the
        # association prior and reported object center across the whole room.
        return np.percentile(points, [trim_percentile, 100.0 - trim_percentile], axis=0).reshape(6)
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


def iou_3d_matrix(boxes_a: Array, boxes_b: Array) -> Array:
    """Pairwise :func:`iou_3d` between (N, 6) and (M, 6) box arrays, as an (N, M) array."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 6)[:, None, :]
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 6)[None, :, :]
    inter = np.prod(np.clip(np.minimum(a[..., 3:], b[..., 3:]) - np.maximum(a[..., :3], b[..., :3]), 0.0, None), axis=-1)
    vol_a = np.prod(np.clip(a[..., 3:] - a[..., :3], 0.0, None), axis=-1)
    vol_b = np.prod(np.clip(b[..., 3:] - b[..., :3], 0.0, None), axis=-1)
    union = vol_a + vol_b - inter
    return np.where(union > 1e-12, inter / np.where(union > 1e-12, union, 1.0), 0.0)


def bbox3d_gap(bbox3d_a: Array, bbox3d_b: Array) -> float:
    """Euclidean distance between two axis-aligned 3D boxes; 0 when they touch or overlap."""
    a, b = np.asarray(bbox3d_a, dtype=np.float64), np.asarray(bbox3d_b, dtype=np.float64)
    return float(np.linalg.norm(np.clip(np.maximum(a[:3] - b[3:], b[:3] - a[3:]), 0.0, None)))


def overlap_3d(bbox3d_a: Array, bbox3d_b: Array, pad: float = 0.0) -> float:
    """Intersection volume over the smaller box's volume, with every side grown by ``pad``.

    1 when one box contains the other, whatever their sizes -- unlike IoU, a
    partial view of an object inside the object's full box scores 1. The pad
    keeps a flat box (one voxel thick) from having zero volume.
    """
    a = np.asarray(bbox3d_a, dtype=np.float64) + np.array([-pad] * 3 + [pad] * 3)
    b = np.asarray(bbox3d_b, dtype=np.float64) + np.array([-pad] * 3 + [pad] * 3)
    inter = float(np.prod(np.clip(np.minimum(a[3:], b[3:]) - np.maximum(a[:3], b[:3]), 0.0, None)))
    smaller = min(float(np.prod(np.clip(a[3:] - a[:3], 0.0, None))), float(np.prod(np.clip(b[3:] - b[:3], 0.0, None))))
    return inter / smaller if smaller > 1e-12 else 0.0


def centroid(bbox3d: Array) -> Array:
    """Center of an axis-aligned 3D box [xmin, ymin, zmin, xmax, ymax, zmax]."""
    return np.array([
        (bbox3d[0] + bbox3d[3]) / 2.0,
        (bbox3d[1] + bbox3d[4]) / 2.0,
        (bbox3d[2] + bbox3d[5]) / 2.0,
    ])
