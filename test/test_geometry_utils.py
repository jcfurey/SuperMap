import numpy as np

from semantic_mapping.geometry_utils import (
    bbox3d_from_points,
    centroid,
    depth_consistency_mask,
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
