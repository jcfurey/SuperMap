"""Full-cloud coverage, temporal map semantics, and evidence boundaries."""
import numpy as np
import pytest

from semantic_mapping.dense_cloud import CameraLabels, DenseCloudConfig, DenseCloudPipeline, DIRECT, PROPAGATED
from semantic_mapping.types import CameraIntrinsics, Detection2D


def make_pipeline(**kwargs):
    return DenseCloudPipeline(DenseCloudConfig(min_region_voxels=1, **kwargs))


def camera(stamp=1., mask=None, label="chair", **kwargs):
    intr = CameraIntrinsics(10., 10., 5., 5., 10, 10)
    if mask is None:
        mask = np.ones((10, 10), bool)
    detection = Detection2D(np.array([0., 0., 10., 10.]), label, 1., mask=mask)
    return CameraLabels(stamp, intr, np.eye(4), [detection], **kwargs)


def test_full_cloud_preserves_points_outside_camera_behind_occluder_and_invalid_rows():
    xyz = np.array([[0., 0., 2.], [0., 0., 4.], [10., 0., 2.], [0., 0., -2.], [np.nan, 0., 0.]])
    pipeline = make_pipeline()
    geometry = pipeline.update(xyz, 1.)
    assert len(geometry.points_world) == 5
    assert geometry.valid.tolist() == [True, True, True, True, False]
    assert np.all(geometry.semantic_ids == 0)
    result = pipeline.annotate(camera())
    assert result.camera_visible.tolist() == [True, False, False, False, False]
    assert result.semantic_ids.tolist() == [1, 0, 0, 0, 0]
    np.testing.assert_equal(result.points_world, xyz)
    np.testing.assert_array_equal(result.region_ids, geometry.region_ids)
    assert result.region_ids[-1] == -1
    assert result.stats["pretrained_models"] == []


def test_missing_camera_does_not_remove_unseen_regions_or_labels():
    pipeline = make_pipeline(input_mode="scan")
    first = np.array([[0., 0., 2.], [10., 0., 2.]])
    initial = pipeline.update(first, 1.)
    labeled = pipeline.annotate(camera())
    other = pipeline.update(np.array([[20., 0., 2.]]), 2.)
    assert len(other.map_points_world) == 3
    assert np.count_nonzero(other.map_semantic_ids) == 1
    assert labeled.region_ids[0] in other.map_region_ids
    assert initial.region_ids[1] in other.map_region_ids
    assert not other.camera_visible.any()


def test_snapshot_replaces_map_scan_unions_and_reordered_points_keep_region_ids():
    a = np.array([[0., 0., 2.], [1., 0., 2.]])
    for mode, expected in [("snapshot", 1), ("scan", 2)]:
        pipeline = make_pipeline(input_mode=mode)
        initial = pipeline.update(a, 1.)
        reordered = pipeline.update(a[::-1], 2.)
        np.testing.assert_array_equal(reordered.region_ids, initial.region_ids[::-1])
        final = pipeline.update(a[:1], 3.)
        assert len(final.map_points_world) == expected
        assert final.region_ids[0] == initial.region_ids[0]


def test_registered_scan_and_camera_pose_use_world_coordinates():
    pipeline = make_pipeline(input_mode="scan")
    pose = np.eye(4)
    pose[:3, 3] = [12., 2., -1.]
    result = pipeline.update(np.array([[0., 0., 2.]]), 1., pose)
    np.testing.assert_allclose(result.points_world, [[12., 2., 1.]])
    observation = camera()
    observation.T_world_from_camera = pose
    assert pipeline.annotate(observation).semantic_ids.tolist() == [1]


def test_camera_depth_is_positive_visibility_evidence_and_invalid_depth_is_unknown():
    for depth, expected in [(1., 0), (2., 1), (3., 0), (0., 0), (np.nan, 0)]:
        pipeline = make_pipeline()
        pipeline.update(np.array([[0., 0., 2.]]), 1.)
        result = pipeline.annotate(camera(depth=np.full((10, 10), depth)))
        assert result.semantic_ids[0] == expected


def test_camera_mask_coverage_does_not_use_detection_box():
    pipeline = make_pipeline()
    pipeline.update(np.array([[0., 0., 2.], [.4, 0., 2.]]), 1.)
    mask = np.zeros((10, 10), bool)
    mask[5, 5] = True
    result = pipeline.annotate(camera(mask=mask))
    assert result.semantic_ids.tolist() == [1, 0]
    assert result.camera_visible.all()


def test_overlapping_classes_and_conflicting_labels_in_one_voxel_are_not_fused():
    pipeline = make_pipeline()
    pipeline.update(np.array([[0., 0., 2.]]), 1.)
    observation = camera()
    observation.detections += camera(label="table").detections
    assert pipeline.annotate(observation).semantic_ids.tolist() == [0]


def test_distinct_mask_pixels_with_conflicting_classes_in_one_voxel_remain_unknown():
    pipeline = make_pipeline()
    pipeline.update(np.array([[0., 0., 2.], [.005, 0., 2.]]), 1.)
    observation = camera()
    observation.intrinsics.fx = 1000.
    first, second = np.zeros((10, 10), bool), np.zeros((10, 10), bool)
    first[5, 5], second[5, 7] = True, True
    observation.detections = camera(mask=first).detections+camera(mask=second, label="table").detections
    result = pipeline.annotate(observation)
    assert result.camera_visible.all()
    assert result.semantic_ids.tolist() == [0, 0]


@pytest.mark.parametrize("failure", ["stale", "foreign", "box_only", "bad_mask", "bad_pose"])
def test_unusable_camera_input_rejected_without_changing_geometry(failure):
    pipeline = make_pipeline()
    before = pipeline.update(np.array([[0., 0., 2.], [10., 0., 2.]]), 1.)
    observation = camera()
    if failure == "stale":
        observation.stamp = 0.
    elif failure == "foreign":
        observation.source = "unapproved_model"
    elif failure == "box_only":
        observation.detections[0].mask = None
    elif failure == "bad_mask":
        observation.detections[0].mask = np.ones((3, 4), bool)
    else:
        observation.T_world_from_camera[0, 0] = 2.
    with pytest.raises(ValueError):
        pipeline.annotate(observation)
    after = pipeline.result()
    np.testing.assert_array_equal(after.points_world, before.points_world)
    np.testing.assert_array_equal(after.semantic_ids, before.semantic_ids)


def test_explicit_capacity_error_keeps_previous_map_instead_of_silent_truncation():
    pipeline = make_pipeline(input_mode="scan", max_map_voxels=1)
    pipeline.update(np.array([[0., 0., 2.]]), 1.)
    with pytest.raises(ValueError, match="no points were silently cropped"):
        pipeline.update(np.array([[1., 0., 2.]]), 2.)
    assert pipeline.stamp == 1.
    assert len(pipeline.result().map_points_world) == 1


def test_one_voxel_keeps_all_original_dense_point_rows():
    pipeline = make_pipeline(voxel_size=.1)
    cloud = np.tile([.03, .03, 2.03], (5000, 1))
    result = pipeline.update(cloud, 1.)
    assert len(result.points_world) == 5000 and len(result.map_points_world) == 1
    assert len(np.unique(result.region_ids)) == 1


def test_propagation_is_opt_in_bounded_same_region_and_never_reseeds_itself():
    xyz = np.array([[x, 0., 2.] for x in np.arange(0., 1.1, .1)])
    pipeline = make_pipeline(voxel_size=.04, neighbor_radius=.15, label_propagation_radius=.25)
    pipeline.update(xyz, 1.)
    mask = np.zeros((10, 10), bool)
    mask[5, 5] = True
    result = pipeline.annotate(camera(mask=mask))
    direct = result.label_source == DIRECT
    propagated = result.label_source == PROPAGATED
    assert direct.any() and propagated.any() and (result.label_source == 0).any()
    distances = np.abs(xyz[propagated, 0, None]-xyz[direct, 0]).min(axis=1)
    assert np.all(distances <= .25)
    again = pipeline.result()
    np.testing.assert_array_equal(again.label_source, result.label_source)
    assert np.count_nonzero(pipeline.semantic) == direct.sum()


def test_empty_cloud_modes_and_clock_rewind():
    for mode, count in [("snapshot", 0), ("scan", 1)]:
        pipeline = make_pipeline(input_mode=mode)
        pipeline.update(np.array([[0., 0., 2.]]), 1.)
        result = pipeline.update(np.empty((0, 3)), 2.)
        assert len(result.points_world) == 0 and len(result.map_points_world) == count
        with pytest.raises(ValueError, match="nondecreasing"):
            pipeline.update(np.empty((0, 3)), 1.)


def test_label_expiry_does_not_expire_geometry():
    pipeline = make_pipeline(label_ttl_sec=.5)
    xyz = np.array([[0., 0., 2.]])
    pipeline.update(xyz, 1.)
    pipeline.annotate(camera())
    result = pipeline.update(xyz, 2.)
    assert result.semantic_ids.tolist() == [0] and result.region_ids[0] > 0
