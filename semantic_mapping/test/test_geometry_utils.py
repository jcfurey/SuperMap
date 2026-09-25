import numpy as np

from semantic_mapping.geometry_utils import (
    back_project_depth,
    bbox3d_from_points,
    centroid,
    depth_consistency_mask,
    foreground_depth_mask,
    invert_se3,
    iou_xy,
    iou_xyxy,
    project_point,
    quaternion_to_rotation_matrix,
    rasterize_depth,
    rotation_matrix_to_quaternion,
    se3_from_translation_quaternion,
    transform_points,
)


def test_robust_bbox_rejects_sparse_depth_outliers_without_changing_default_bounds():
    points = np.random.default_rng(2).uniform(0, 1, (1000, 3))
    points = np.vstack([points, [65.535, 30.0, 15.0]])
    full = bbox3d_from_points(points)
    robust = bbox3d_from_points(points, trim_percentile=2.0)
    np.testing.assert_allclose(full[3:], [65.535, 30.0, 15.0])
    assert np.all(robust[:3] >= 0) and np.all(robust[3:] <= 1)
    assert np.all(robust[3:] - robust[:3] > .9)


def test_early_backprojection_sampling_preserves_the_filtered_point_sample():
    rng = np.random.default_rng(12)
    depth = rng.uniform(1.8, 2.2, (120, 160))
    depth[::8, ::8] = 5.0
    depth[::9, ::9] = np.nan
    mask = rng.random(depth.shape) > .3
    K = np.array([[100., 0, 80], [0, 100., 60], [0, 0, 1.]])
    full = back_project_depth(K, depth, mask)
    full = full[depth_consistency_mask(full[:, 2])]
    indices = np.random.default_rng(0).choice(len(full), size=400, replace=False)
    actual = back_project_depth(K, depth, mask, max_points=400, depth_mad_factor=3)
    np.testing.assert_array_equal(actual, full[indices])


def test_invert_se3_round_trip():
    T = se3_from_translation_quaternion(
        np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3, np.sqrt(1 - 0.14)]),
    )
    identity = T @ invert_se3(T)
    assert np.allclose(identity, np.eye(4), atol=1e-8)


def test_quaternion_round_trip():
    q = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])  # 90 deg about z
    R = quaternion_to_rotation_matrix(*q)
    q2 = rotation_matrix_to_quaternion(R)
    R2 = quaternion_to_rotation_matrix(*q2)
    assert np.allclose(R, R2, atol=1e-8)


def test_project_point_matches_pinhole_model():
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    T_world_from_cam = np.eye(4)
    point_world = np.array([1.0, 0.5, 2.0])
    pixel, depth = project_point(K, invert_se3(T_world_from_cam), point_world)
    assert depth == 2.0
    assert np.allclose(pixel, [100 * 1.0 / 2.0 + 50.0, 100 * 0.5 / 2.0 + 40.0])


def test_transform_points_translation_only():
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    out = transform_points(T, points)
    assert np.allclose(out, [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])


def test_iou_xyxy_perfect_overlap_and_disjoint():
    box = np.array([0.0, 0.0, 10.0, 10.0])
    assert iou_xyxy(box, box) == 1.0
    disjoint = np.array([20.0, 20.0, 30.0, 30.0])
    assert iou_xyxy(box, disjoint) == 0.0


def test_iou_xyxy_partial_overlap():
    a = np.array([0.0, 0.0, 10.0, 10.0])
    b = np.array([5.0, 5.0, 15.0, 15.0])
    # intersection 5x5=25, union 100+100-25=175
    assert np.isclose(iou_xyxy(a, b), 25.0 / 175.0)


def test_bbox3d_and_centroid():
    points = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    box = bbox3d_from_points(points)
    assert np.allclose(box, [0.0, 0.0, 0.0, 2.0, 4.0, 6.0])
    assert np.allclose(centroid(box), [1.0, 2.0, 3.0])


def test_iou_3d():
    from semantic_mapping.geometry_utils import iou_3d

    a = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0])
    assert iou_3d(a, a) == 1.0
    assert iou_3d(a, np.array([5.0, 5.0, 5.0, 6.0, 6.0, 6.0])) == 0.0
    b = np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0])  # intersection 1, union 8+8-1
    assert np.isclose(iou_3d(a, b), 1.0 / 15.0)


def test_iou_xy_on_stacked_boxes():
    lower = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.5])
    upper = np.array([0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
    assert iou_xy(lower, upper) == 1.0


def test_rasterize_depth_nearest_point_wins():
    K = np.eye(3)
    K[0, 0] = K[1, 1] = 1.0
    K[0, 2] = K[1, 2] = 0.0
    points_cam = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 2.0]])
    depth = rasterize_depth(points_cam, K, width=1, height=1)
    assert depth[0, 0] == 2.0


def test_depth_consistency_mask_rejects_outlier():
    depths = np.array([2.0, 2.05, 1.95, 2.1, 8.0])
    mask = depth_consistency_mask(depths)
    assert mask.tolist() == [True, True, True, True, False]


def test_depth_consistency_mask_empty():
    assert depth_consistency_mask(np.zeros(0)).shape == (0,)


def test_foreground_layer_survives_majority_background_and_near_speckles():
    # The median belongs to the wall; the sparse person is the first supported layer.
    depths = np.concatenate(([1.0, 1.1, 0.0, np.nan, np.inf, -2.0],
                             np.linspace(12.0, 12.3, 8), np.linspace(30, 30.4, 50)))
    selected = depths[foreground_depth_mask(depths, .75, 5, .1)]
    np.testing.assert_array_equal(selected, depths[6:14])


def test_foreground_layer_requires_sensor_support():
    for depths in [np.array([]), np.array([0, np.nan]), np.array([2., 2.1, 10., 10.1])]:
        assert not foreground_depth_mask(depths, .75, 5, .1).any()


def test_foreground_layer_preserves_input_order_and_single_surface():
    depths = np.array([4.2, 4.0, 4.3, 4.1, 4.4])
    np.testing.assert_array_equal(depths[foreground_depth_mask(depths, .75)], depths)


def test_foreground_layer_requires_spatial_support_not_just_many_foot_returns():
    feet = np.column_stack((np.linspace(0, 30, 30), np.full(30, 39)))
    x, y = np.meshgrid(np.arange(0, 31, 5), np.arange(0, 40, 5))
    body = np.column_stack((x.ravel(), y.ravel()))
    pixels = np.vstack((feet, body))
    depths = np.concatenate((np.full(len(feet), 2.), np.full(len(body), 3.)))
    keep = foreground_depth_mask(depths, .75, pixels=pixels, min_span=np.array([15, 20]))
    assert not keep[:len(feet)].any() and keep[len(feet):].all()
    assert not foreground_depth_mask(depths[:len(feet)], .75, pixels=feet, min_span=np.array([15, 20])).any()


def test_fill_sparse_depth_fills_only_empty_pixels_with_neighbourhood_minimum():
    from semantic_mapping.geometry_utils import fill_sparse_depth

    depth = np.zeros((5, 5))
    depth[1, 1] = 4.0
    depth[3, 3] = 2.0
    filled = fill_sparse_depth(depth, radius_px=1)
    assert filled[1, 1] == 4.0 and filled[3, 3] == 2.0          # readings untouched
    assert filled[0, 0] == 4.0 and filled[2, 2] == 2.0          # nearest neighbours, minimum where both reach
    assert filled[4, 0] == 0.0 and filled[0, 4] == 0.0          # nobody within radius 1
    assert np.array_equal(fill_sparse_depth(depth, radius_px=0), depth)
    assert fill_sparse_depth(depth, radius_px=3)[4, 0] == 2.0   # wider radius reaches, and takes the minimum
    dense = np.full((3, 3), 1.5)
    assert np.array_equal(fill_sparse_depth(dense, 2), dense)


def test_fill_sparse_depth_equals_a_square_minimum_filter_over_valid_readings():
    from scipy.ndimage import minimum_filter

    from semantic_mapping.geometry_utils import fill_sparse_depth

    rng = np.random.default_rng(3)
    for shape in [(1, 1), (1, 9), (9, 1), (31, 17), (120, 160)]:
        for density in (0.02, 0.3):
            for radius in (1, 2, 5):
                depth = np.where(rng.random(shape) < density, rng.uniform(0.1, 50.0, shape), 0.0)
                special = rng.random(shape) < 0.05
                depth[special] = rng.choice([np.nan, np.inf, -np.inf, -1.0], special.sum())
                valid = np.isfinite(depth) & (depth > 0)
                expected = minimum_filter(np.where(valid, depth, np.inf), size=2 * radius + 1, mode="nearest")
                expected = np.where(valid, depth, expected)
                expected[~np.isfinite(expected)] = 0.0
                before = depth.copy()
                actual = fill_sparse_depth(depth, radius)
                assert actual.dtype == np.float64
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(depth, before)  # the input is not modified
    # No reading at all: nothing to fill from, every pixel stays invalid.
    np.testing.assert_array_equal(fill_sparse_depth(np.full((4, 4), np.nan), 2), np.zeros((4, 4)))


def test_mask_bounds():
    from semantic_mapping.geometry_utils import mask_bounds

    mask = np.zeros((6, 8), bool)
    assert mask_bounds(mask) is None
    mask[2, 3] = mask[4, 6] = True
    assert mask_bounds(mask) == (2, 5, 3, 7)
    mask[:] = True
    assert mask_bounds(mask) == (0, 6, 0, 8)


def test_back_projecting_a_crop_at_its_pixel_origin_matches_the_full_image():
    rng = np.random.default_rng(5)
    depth = rng.uniform(1.0, 4.0, (120, 160))
    depth[rng.random(depth.shape) < .3] = 0.0
    mask = np.zeros(depth.shape, bool)
    mask[40:90, 30:110] = rng.random((50, 80)) > .2
    K = np.array([[101.3, 0, 79.37], [0, 99.1, 61.83], [0, 0, 1.]])  # non-integer principal point
    for kwargs in ({}, {"max_points": 500, "depth_mad_factor": 3.0}):
        full = back_project_depth(K, depth, mask, **kwargs)
        crop = back_project_depth(K, depth[35:95, 25:120], mask[35:95, 25:120], pixel_origin=(25, 35), **kwargs)
        np.testing.assert_array_equal(crop, full)


def test_rasterize_depth_keeps_the_nearest_point_per_pixel():
    from semantic_mapping.geometry_utils import rasterize_depth

    K = np.array([[100.0, 0.0, 8.0], [0.0, 100.0, 6.0], [0.0, 0.0, 1.0]])
    rng = np.random.default_rng(0)
    points = rng.uniform([-0.1, -0.1, 0.5], [0.1, 0.1, 5.0], size=(2000, 3))  # many points per pixel
    points = np.concatenate([points, [[0.0, 0.0, -1.0], [50.0, 0.0, 1.0]]])   # behind the camera, off-image
    depth = rasterize_depth(points, K, 16, 12)

    z = points[:, 2]
    front = z > 0
    us = np.round(100.0 * points[front, 0] / z[front] + 8.0).astype(int)
    vs = np.round(100.0 * points[front, 1] / z[front] + 6.0).astype(int)
    expected = np.zeros((12, 16))
    for u, v, d in sorted(zip(us, vs, z[front]), key=lambda t: -t[2]):  # far-to-near reference
        if 0 <= u < 16 and 0 <= v < 12:
            expected[v, u] = d
    assert np.array_equal(depth, expected)
    assert rasterize_depth(np.zeros((0, 3)), K, 16, 12).shape == (12, 16)


def test_packed_voxel_keys_order_like_the_cells_and_refuse_out_of_range():
    from semantic_mapping.geometry_utils import pack_voxel_keys

    limit = 1 << 20
    cells = np.array([[-limit, 5, 0], [-1, -1, -1], [0, 0, 0], [0, 0, 1], [0, 1, -limit], [limit - 1, limit - 1, limit - 1]])
    keys = pack_voxel_keys(cells)
    assert keys.dtype == np.int64 and np.all(np.diff(keys) > 0)  # rows above are in lexicographic order
    assert pack_voxel_keys(np.array([[limit, 0, 0]])) is None
    assert pack_voxel_keys(np.array([[0, -limit - 1, 0]])) is None
    assert pack_voxel_keys(np.zeros((0, 3), np.int64)).shape == (0,)
