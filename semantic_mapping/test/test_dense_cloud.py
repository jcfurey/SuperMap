"""Full-cloud coverage, temporal map semantics, and evidence boundaries."""
import numpy as np
import pytest

from semantic_mapping.dense_cloud import CameraLabels, DenseCloudConfig, DenseCloudPipeline, DIRECT, PROPAGATED
from semantic_mapping.types import CameraIntrinsics, Detection2D


def make_pipeline(**kwargs):
    kwargs.setdefault("min_region_voxels", 1)
    return DenseCloudPipeline(DenseCloudConfig(**kwargs))


def test_yoloe_exception_is_explicit_and_does_not_label_unseen_geometry():
    points = np.array([[0., 0., 2.], [10., 0., 2.]])
    strict = make_pipeline()
    strict.update(points, 1.)
    with pytest.raises(ValueError, match="explicitly allowed"):
        strict.annotate(camera(source="yoloe"))
    allowed = make_pipeline(allow_yoloe_labels=True)
    allowed.update(points, 1.)
    result = allowed.annotate(camera(source="yoloe"))
    assert result.semantic_ids.tolist() == [1, 0]
    assert result.stats["annotation_sources"] == ["yoloe"]
    np.testing.assert_array_equal(result.points_world, points)
    with pytest.raises(ValueError, match="explicitly allowed"):
        allowed.annotate(camera(source="another_model"))


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
    pipeline.update(np.array([[0., 0., 2.], [.004, 0., 2.]]), 1.)  # pixels 5 and 7, one voxel
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


def _person_and_wall(splat=True, layer=True):
    """A sparse person (0.1 m returns at 3 m) in front of a denser wall at 20 m."""
    intr = CameraIntrinsics(100., 100., 50., 50., 100, 100)
    ys = np.arange(-0.9, 0.91, 0.1)
    xs = np.arange(-0.3, 0.31, 0.1)
    person = np.array([[x, y, 3.] for x in xs for y in ys])
    wall = np.array([[x, y, 20.] for x in np.arange(-8., 8., 0.2) for y in np.arange(-8., 8., 0.2)])
    config = {}
    if not splat:
        config["camera_splat_radius"] = 0.
    if not layer:
        config["mask_depth_gap"] = 0.
    pipeline = make_pipeline(voxel_size=.02, **config)
    xyz = np.vstack([person, wall])
    pipeline.update(xyz, 1.)
    # The person's silhouette, as a segmenter would produce it: a solid mask.
    mask = np.zeros((100, 100), bool)
    mask[int(50-100*0.9/3)-1:int(50+100*0.9/3)+2, int(50-100*0.3/3)-1:int(50+100*0.3/3)+2] = True
    detection = Detection2D(np.array([0., 0., 100., 100.]), "person", .9, mask=mask)
    result = pipeline.annotate(CameraLabels(1., intr, np.eye(4), [detection]))
    return result, len(person)


def test_sparse_lidar_labels_do_not_bleed_onto_background_behind_a_mask():
    result, n_person = _person_and_wall()
    person, wall = result.semantic_ids[:n_person], result.semantic_ids[n_person:]
    assert (person == 1).mean() > 0.9
    assert not wall.any(), f"{np.count_nonzero(wall)} wall points took the person label"


@pytest.mark.parametrize("splat, layer", [(True, False), (False, True)])
def test_each_bleed_guard_alone_blocks_the_background(splat, layer):
    result, n_person = _person_and_wall(splat=splat, layer=layer)
    assert not result.semantic_ids[n_person:].any()
    assert (result.semantic_ids[:n_person] == 1).mean() > 0.9


def test_without_bleed_guards_the_background_would_take_the_label():
    result, n_person = _person_and_wall(splat=False, layer=False)
    assert result.semantic_ids[n_person:].any()  # documents the failure mode the guards fix


def test_foreground_layer_ignores_lone_noise_in_front():
    from semantic_mapping.dense_cloud import foreground_layer

    config = DenseCloudConfig()
    depths = np.array([0.5, 3.0, 3.1, 3.2, 3.05, 18., 18.2])
    assert foreground_layer(depths, config).tolist() == [False, True, True, True, True, False, False]
    # A receding surface (ring gaps grow with range) stays one layer.
    floor = np.array([2., 2.5, 3.1, 3.8, 4.7, 5.8])
    assert foreground_layer(floor, config).all()


def test_label_vocabulary_is_bounded():
    pipeline = make_pipeline(max_labels=2)
    pipeline.update(np.array([[0., 0., 2.]]), 1.)
    pipeline.annotate(camera())
    with pytest.raises(ValueError, match="max_labels"):
        pipeline.annotate(camera(label="table"))


def _surface_scene(rng):
    plane = np.column_stack([rng.uniform(0, 6, 9000), rng.uniform(0, 2, 9000), rng.normal(0, .002, 9000)])
    box = np.column_stack([rng.uniform(.5, .8, 800), rng.uniform(.5, .8, 800), rng.uniform(.3, .6, 800)])
    return np.vstack([plane, box])


def test_incremental_segmentation_matches_a_full_rebuild():
    from semantic_mapping.dense_cloud import segment_surfaces

    rng = np.random.default_rng(3)
    scene = _surface_scene(rng)
    incremental = make_pipeline(input_mode="scan", voxel_size=.03, min_region_voxels=3)
    full = make_pipeline(input_mode="scan", voxel_size=.03, min_region_voxels=3, incremental_segmentation=False)
    chunks = np.array_split(scene[rng.permutation(len(scene))], 6)
    # Local scans: each one adds a spatially limited patch; the last one touches everything.
    modes = []
    x = scene[:, 0]
    for stamp, patch in enumerate([scene[x < 3], scene[(x >= 3) & (x < 4)], scene[(x >= 4) & (x < 5)],
                                   scene[x >= 5], chunks[0]], start=1):
        a = incremental.update(patch, float(stamp))
        b = full.update(patch, float(stamp))
        np.testing.assert_array_equal(a.map_points_world, b.map_points_world)
        # Same partition (region IDs may be numbered differently).
        pairs = np.unique(np.column_stack([a.map_region_ids, b.map_region_ids]), axis=0)
        assert len(pairs) == len(np.unique(a.map_region_ids)) == len(np.unique(b.map_region_ids))
        modes.append(incremental.last_segmentation["incremental"])
    assert modes == [False, True, True, True, False]  # empty start and a map-wide scan rebuild fully
    fresh = segment_surfaces(incremental.points, incremental.config)
    pairs = np.unique(np.column_stack([fresh, incremental.regions]), axis=0)
    assert len(pairs) == len(np.unique(fresh))


def test_snapshot_with_unchanged_geometry_reuses_the_whole_graph():
    rng = np.random.default_rng(4)
    scene = _surface_scene(rng)
    pipeline = make_pipeline(voxel_size=.03)
    first = pipeline.update(scene, 1.)
    second = pipeline.update(scene, 2.)
    assert pipeline.last_segmentation == {"incremental": True, "dirty_voxels": 0}
    np.testing.assert_array_equal(first.region_ids, second.region_ids)


def test_packed_cell_keys_give_row_wise_unique_and_match_results():
    from semantic_mapping.dense_cloud import _cell_keys, _cell_match, _unique_cells

    rng = np.random.default_rng(4)
    limit = 1 << 20
    for low, high in [(-3, 3), (-limit, limit), (-limit - 5, 5), (0, limit + 5)]:  # last two: fallback
        cells = rng.integers(low, high, (500, 3))
        cells[250:] = cells[:250]  # duplicates
        expected = np.unique(cells, axis=0, return_index=True, return_inverse=True)
        actual = _unique_cells(cells, return_index=True, return_inverse=True)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b.reshape(a.shape))
        np.testing.assert_array_equal(_unique_cells(cells), expected[0])
        old, new = expected[0], np.unique(np.concatenate([expected[0][::2], rng.integers(low, high, (50, 3))]), axis=0)
        index = np.searchsorted(_cell_keys(old), _cell_keys(new))
        found = np.flatnonzero(index < len(old))
        found = found[_cell_keys(old)[index[found]] == _cell_keys(new)[found]]
        matched = _cell_match(old, new)
        np.testing.assert_array_equal(matched[0], index[found])
        np.testing.assert_array_equal(matched[1], found)
    empty = _unique_cells(np.zeros((0, 3), np.int64), return_index=True, return_inverse=True)
    assert [a.shape for a in empty] == [(0, 3), (0,), (0,)]


def test_packed_pairs_give_row_wise_unique_rows_and_counts():
    from semantic_mapping.dense_cloud import _unique_pairs

    rng = np.random.default_rng(9)
    for high in (3, 1000, 2 ** 31 - 1):
        first = rng.integers(0, 500_000, 2000)
        second = rng.integers(0, high, 2000).astype(np.int32)
        first[1000:], second[1000:] = first[:1000], second[:1000]
        rows, counts = np.unique(np.column_stack([first, second]), axis=0, return_counts=True)
        actual_rows, actual_counts = _unique_pairs(first, second, return_counts=True)
        np.testing.assert_array_equal(actual_rows, rows)
        np.testing.assert_array_equal(actual_counts, counts)
        np.testing.assert_array_equal(_unique_pairs(first, second), rows)
    assert _unique_pairs(np.zeros(0, np.int64), np.zeros(0, np.int32)).shape == (0, 2)


def test_workers_is_validated_and_does_not_change_results():
    for bad in (0, -2, True, 1.5):
        with pytest.raises(ValueError, match="workers"):
            DenseCloudConfig(workers=bad)
    rng = np.random.default_rng(2)
    cloud = np.column_stack([rng.uniform(0, 4, 3000), rng.uniform(0, 4, 3000), rng.normal(0, .005, 3000)])
    cloud[:1000, 2] = rng.uniform(0, 1, 1000)
    results = []
    for workers in (1, 2, -1):
        pipeline = make_pipeline(workers=workers, label_propagation_radius=.2)
        pipeline.update(cloud, 1.)
        results.append(pipeline.update(cloud + [[.01, 0, 0]], 2.))
    for other in results[1:]:
        np.testing.assert_array_equal(other.region_ids, results[0].region_ids)
        np.testing.assert_array_equal(other.map_region_ids, results[0].map_region_ids)
