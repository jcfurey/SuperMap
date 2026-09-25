import numpy as np
import pytest

from semantic_mapping.pipeline import PipelineConfig, SemanticMappingPipeline
from semantic_mapping.types import CameraIntrinsics, Detection2D, ObjectStatus, Observation, StampedPose

K = np.array([[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]])
INTRINSICS = CameraIntrinsics(fx=100.0, fy=100.0, cx=80.0, cy=60.0, width=160, height=120)


def _depth_with_object_plane(distance: float) -> np.ndarray:
    depth = np.full((120, 160), 8.0)  # background wall
    depth[40:80, 60:100] = distance  # a ~1m x 1m object roughly centered in frame
    return depth


def _observation(stamp: float, distance: float, with_detection: bool) -> Observation:
    detections = []
    if with_detection:
        detections = [Detection2D(bbox=np.array([60.0, 40.0, 100.0, 80.0]), label="chair", score=0.9)]
    return Observation(
        stamp=stamp,
        pose=StampedPose(stamp=stamp, T_world_from_frame=np.eye(4)),
        intrinsics=INTRINSICS,
        depth=_depth_with_object_plane(distance),
        detections=detections,
    )


def test_static_object_becomes_and_stays_active():
    pipeline = SemanticMappingPipeline(PipelineConfig(min_hits_to_confirm=2))
    for i in range(5):
        result = pipeline.process_frame(_observation(stamp=i * 0.1, distance=2.0, with_detection=True))

    active = [o for o in result.objects if o.status.value == "active"]
    assert len(active) == 1
    assert active[0].label == "chair"

    record = pipeline_json_record(result)
    assert record["label"] == "chair"
    assert record["status"] == "active"


def test_geometry_only_frames_do_not_expire_a_tentative_object_between_detections():
    pipeline = SemanticMappingPipeline()
    for i in range(16):
        obs = _observation(i / 15, 2.0, i in (0, 15))
        obs.detections_evaluated = i in (0, 15)
        result = pipeline.process_frame(obs)
        assert len(result.objects) == 1
        assert result.objects[0].status == (ObjectStatus.ACTIVE if i == 15 else ObjectStatus.TENTATIVE)
    assert result.objects[0].hits == 2
    assert result.objects[0].missed_detection_frames == 0


def test_depth_limits_reject_saturated_geometry_without_false_removal_evidence():
    pipeline = SemanticMappingPipeline(PipelineConfig(max_depth_m=6.0))
    for i in range(2):
        pipeline.process_frame(_observation(i * .1, 2.0, True))
    # Far/saturated depth means no measurement; it cannot prove the chair is gone.
    for i in range(2, 8):
        obs = _observation(i * .1, 65.535, True)
        obs.detections[0].mask = np.zeros((120, 160), dtype=bool)
        obs.detections[0].mask[40:80, 60:100] = True
        result = pipeline.process_frame(obs)
        assert obs.depth[50, 70] == pytest.approx(65.535)  # caller input preserved
        assert len(result.objects) == 1 and result.objects[0].status == ObjectStatus.ACTIVE
        assert np.max(result.objects[0].points_world[:, 2]) <= 2.001
    # Valid free-space evidence still detects actual removal.
    for i in range(8, 20):
        result = pipeline.process_frame(_observation(i * .1, 5.0, False))
    assert result.objects[0].status == ObjectStatus.DISAPPEARED


@pytest.mark.parametrize('params', [
    {'min_depth_m': -1}, {'max_depth_m': float('nan')},
    {'min_depth_m': 2, 'max_depth_m': 1}, {'bbox_trim_percentile': 50},
    {'foreground_depth_gap_m': -1}, {'foreground_depth_gap_m': float('nan')},
    {'foreground_depth_min_points': 0}, {'foreground_depth_min_points': 1.5},
    {'foreground_depth_min_fraction': -0.1}, {'foreground_depth_min_fraction': float('nan')},
    {'foreground_depth_min_image_span': 1.1}, {'foreground_depth_min_image_span': float('nan')},
    {'foreground_depth_max_extent_m': -1.}, {'foreground_depth_max_extent_m': float('inf')},
    {'dynamic_geometry_min_extent_fraction': -0.1},
])
def test_invalid_geometry_limits_are_rejected(params):
    with pytest.raises(ValueError):
        PipelineConfig(**params)


def test_mask_depth_gate_rejects_background_speckles_without_changing_mask():
    obs = _observation(0, 2, True)
    mask = np.zeros(obs.depth.shape, dtype=bool)
    mask[40:80, 60:100] = True
    obs.detections[0].mask = mask
    obs.depth[45:47, 65:67] = 5.0  # a few mismatched depth pixels inside a correct mask
    original_mask = mask.copy()
    raw = SemanticMappingPipeline().process_frame(obs).objects[0]
    filtered = SemanticMappingPipeline(PipelineConfig(mask_depth_mad_factor=3)).process_frame(obs).objects[0]
    assert raw.bbox3d[5] > 4.9
    assert filtered.bbox3d[5] < 2.1
    np.testing.assert_array_equal(mask, original_mask)


def test_foreground_depth_selection_precedes_filling_and_preserves_other_classes():
    depth = np.zeros((120, 160))
    depth[45:75:3, 65:95:3] = 25.0  # most returns inside the person mask hit the far wall
    depth[50:70:3, 78] = 8.0
    depth[47, 69] = 1.0  # isolated foreground noise must not expand into a supported layer
    mask = np.zeros(depth.shape, dtype=bool)
    mask[40:80, 60:100] = True
    original = depth.copy()
    config = PipelineConfig(foreground_depth_gap_m=.75, foreground_depth_min_fraction=.05,
                            depth_fill_radius_px=2)
    pipeline = SemanticMappingPipeline(config)
    detection = Detection2D(bbox=np.array([60., 40., 100., 80.]), label='person', score=.9, mask=mask)
    points = pipeline._detection_points_world(detection, depth, K, np.eye(4))
    assert len(points) > 7  # filling still provides usable foreground geometry
    np.testing.assert_allclose(points[:, 2], 8.0)
    np.testing.assert_array_equal(depth, original)
    assert mask.sum() == 1600
    detection.label = 'pipe'  # extended objects retain their full depth range
    points = pipeline._detection_points_world(detection, depth, K, np.eye(4))
    assert points[:, 2].min() == 1.0 and points[:, 2].max() == 25.0


def test_foreground_depth_does_not_invent_geometry_from_unsupported_returns():
    obs = _observation(0, 2, True)
    obs.depth[:] = 0
    obs.depth[60, 80] = 2
    obs.detections[0].label = 'person'
    pipeline = SemanticMappingPipeline(PipelineConfig(foreground_depth_gap_m=.75, depth_fill_radius_px=3))
    obj = pipeline.process_frame(obs).objects[0]
    assert obj.points_world.shape == (0, 3)


def _person_observation(stamp, distance=3.):
    obs = _observation(stamp, distance, True)
    obs.detections[0].label = 'person'
    obs.detections[0].mask = np.zeros(obs.depth.shape, dtype=bool)
    obs.detections[0].mask[40:80,60:100] = True
    return obs


def test_partial_person_depth_preserves_last_body_and_geometry_time():
    p = SemanticMappingPipeline(PipelineConfig(dynamic_geometry_enabled=True, foreground_depth_gap_m=.75,
                                               foreground_depth_min_image_span=.5, depth_fill_radius_px=2))
    p.process_frame(_person_observation(0))
    obj = p.process_frame(_person_observation(.1)).objects[0]
    old_points, old_evidence = obj.points_world.copy(), obj.point_log_odds.copy()
    for i in range(2, 8):
        obs = _person_observation(i*.1)
        obs.depth[:] = 0
        obs.depth[75:80,60:100] = 2.5  # many real points, but only at the feet
        p.process_frame(obs)
        assert obj.status == ObjectStatus.OCCLUDED
        assert obj.geometry_stamp == .1 and obj.latest_stamp == i*.1
        np.testing.assert_array_equal(obj.points_world, old_points)
        np.testing.assert_array_equal(obj.point_log_odds, old_evidence)
    p.process_frame(_person_observation(.8, 2.5))
    assert obj.status == ObjectStatus.ACTIVE and obj.geometry_stamp == .8 and obj.instance_id == 1


def test_person_extent_guard_rejects_background_without_clipping_or_affecting_other_classes():
    p = SemanticMappingPipeline(PipelineConfig(foreground_depth_gap_m=.75, foreground_depth_min_image_span=.5,
                                               foreground_depth_max_extent_m=3.))
    obs = _person_observation(0, 30.)
    d = obs.detections[0]
    assert len(p._detection_points_world(d, obs.depth, K, np.eye(4))) == 0
    d.label = 'pipe'
    points = p._detection_points_world(d, obs.depth, K, np.eye(4))
    assert len(points) and np.ptp(points[:,1]) > 10.


def test_depthless_person_keeps_one_2d_track_without_inventing_a_mapped_location():
    p = SemanticMappingPipeline(PipelineConfig(dynamic_geometry_enabled=True, foreground_depth_gap_m=.75))
    for i in range(3):
        obs = _person_observation(i*.1, 0.)
        obs.pose.T_world_from_frame[:3,3] = [5, 0, -5]
        result = p.process_frame(obs)
        assert len(result.objects) == 1
        obj = result.objects[0]
        assert obj.status == ObjectStatus.TENTATIVE and obj.instance_id == 1 and not len(obj.points_world)
        assert not result.scene_graph.node_ids
    obs = _person_observation(.3)
    obs.pose.T_world_from_frame[:3,3] = [5, 0, -5]
    obj = p.process_frame(obs).objects[0]
    assert obj.instance_id == 1 and obj.status == ObjectStatus.ACTIVE and obj.geometry_stamp == .3


def test_expired_depthless_tracks_do_not_become_spatial_memories_at_the_origin():
    p = SemanticMappingPipeline(PipelineConfig(dynamic_geometry_enabled=True, tentative_max_age=2))
    p.process_frame(_person_observation(0., 0.))
    for i in range(1, 5):
        obs = _person_observation(i*.1, 0.)
        obs.detections = []
        result = p.process_frame(obs)
    assert result.objects[0].status == ObjectStatus.DISAPPEARED
    assert result.scene_graph.node_ids == []


@pytest.mark.parametrize('label, dynamic, replaces', [('person', True, True), ('chair', True, False), ('person', False, False)])
def test_dynamic_geometry_keeps_latest_position_with_identity_and_history(label, dynamic, replaces):
    pipeline = SemanticMappingPipeline(PipelineConfig(dynamic_geometry_enabled=dynamic))
    for i, distance in enumerate([5., 4., 3.]):
        obs = _observation(i*.1, distance, True)
        obs.detections[0].label = label
        result = pipeline.process_frame(obs)
        assert len(result.objects) == 1
    obj = result.objects[0]
    assert obj.instance_id == 1 and obj.hits == 3 and obj.status == ObjectStatus.ACTIVE
    assert len(obj.trajectory) == 3
    assert obj.trajectory[0][1][2] == pytest.approx(5.)
    if replaces:
        np.testing.assert_allclose(obj.points_world[:, 2], 3.)
        assert obj.bbox3d[5]-obj.bbox3d[2] < .1
    else:
        assert obj.points_world[:, 2].max() == 5.
    # No return is unknown: preserve the last supported geometry.
    previous = obj.points_world.copy()
    obs = _observation(.3, 0., True)
    obs.detections[0].label = label
    pipeline.process_frame(obs)
    np.testing.assert_array_equal(obj.points_world, previous)


def test_confirmation_threshold_is_honored_by_matching_and_reidentification():
    pipeline = SemanticMappingPipeline(PipelineConfig(min_hits_to_confirm=5))
    for i in range(2):
        result = pipeline.process_frame(_observation(i * 0.1, 2.0, True))
        assert result.objects[0].status == ObjectStatus.TENTATIVE
    for i in range(2, 8):
        result = pipeline.process_frame(_observation(i * 0.1, 8.0, False))
    assert result.objects[0].status == ObjectStatus.DISAPPEARED
    for i in range(8, 11):
        result = pipeline.process_frame(_observation(i * 0.1, 2.0, True))
        assert len(result.objects) == 1 and result.objects[0].instance_id == 1
        obj = result.objects[0]
        assert obj.status == (ObjectStatus.ACTIVE if obj.hits >= 5 else ObjectStatus.TENTATIVE)
    assert result.objects[0].hits == 5
    for i in range(11, 18):
        result = pipeline.process_frame(_observation(i * 0.1, 8.0, False))
    assert result.objects[0].status == ObjectStatus.DISAPPEARED
    result = pipeline.process_frame(_observation(1.8, 2.0, True))
    assert result.objects[0].instance_id == 1 and result.objects[0].hits == 6
    assert result.objects[0].status == ObjectStatus.ACTIVE  # a confirmed identity resumes immediately


@pytest.mark.parametrize('stamp', [0.0, 0.1])
def test_stale_or_duplicate_frames_cannot_change_the_map(stamp):
    pipeline = SemanticMappingPipeline()
    pipeline.process_frame(_observation(0.0, 2.0, True))
    pipeline.process_frame(_observation(0.1, 2.0, True))
    obj = pipeline.object_map.objects[1]
    points, evidence, hits = obj.points_world.copy(), obj.point_log_odds.copy(), obj.hits
    with pytest.raises(ValueError, match='timestamp order'):
        pipeline.process_frame(_observation(stamp, 8.0, False))
    assert pipeline._frame_index == 2 and obj.hits == hits
    np.testing.assert_array_equal(obj.points_world, points)
    np.testing.assert_array_equal(obj.point_log_odds, evidence)


def test_invalid_masks_are_rejected_before_mutating_the_map():
    pipeline = SemanticMappingPipeline()
    observation = _observation(0.0, 2.0, True)
    observation.detections[0].mask = np.ones((60, 80), dtype=bool)
    with pytest.raises(ValueError, match='boolean mask of shape'):
        pipeline.process_frame(observation)
    assert pipeline._frame_index == 0 and not pipeline.object_map.objects
    observation.detections[0].mask = None
    assert len(pipeline.process_frame(observation).objects) == 1


def test_object_confirmed_disappeared_after_surface_moves_away():
    pipeline = SemanticMappingPipeline(PipelineConfig(min_hits_to_confirm=2, disappeared_occupied_fraction=0.3))
    for i in range(5):
        pipeline.process_frame(_observation(stamp=i * 0.1, distance=2.0, with_detection=True))

    # Object stops being detected and the depth surface behind it moves far away,
    # i.e. it was physically removed.
    result = None
    for i in range(5, 5 + 40):
        result = pipeline.process_frame(_observation(stamp=i * 0.1, distance=8.0, with_detection=False))

    statuses = {o.status.value for o in result.objects}
    assert "disappeared" in statuses


def pipeline_json_record(result):
    from semantic_mapping.serialization import serialize_frame

    records = serialize_frame(result.objects, result.scene_graph)
    return next(r for r in records if r["status"] == "active")


def test_reactivation_preserves_identity_when_2d_prediction_is_off():
    pipeline = SemanticMappingPipeline(PipelineConfig(min_hits_to_confirm=2))
    for i in range(4):
        pipeline.process_frame(_observation(stamp=i * 0.1, distance=2.0, with_detection=True))
    (only_id,) = [o.instance_id for o in pipeline.object_map.objects.values()]

    # Same physical object plane, but the detection box is shifted 25 px so its
    # 2D IoU with the predicted box falls below the association threshold.
    depth = np.full((120, 160), 8.0)
    depth[40:80, 60:125] = 2.0  # widen the plane so the shifted box still lands on it
    shifted = Observation(
        stamp=0.4,
        pose=StampedPose(stamp=0.4, T_world_from_frame=np.eye(4)),
        intrinsics=INTRINSICS,
        depth=depth,
        detections=[Detection2D(bbox=np.array([85.0, 40.0, 125.0, 80.0]), label="chair", score=0.9)],
    )
    result = pipeline.process_frame(shifted)

    ids = sorted(o.instance_id for o in result.objects)
    assert ids == [only_id], f"expected a single re-activated identity, got {ids}"
    assert result.objects[0].status.value == "active"


def test_low_confidence_detection_does_not_spawn_new_object():
    pipeline = SemanticMappingPipeline(PipelineConfig(high_score_threshold=0.5))
    obs = _observation(stamp=0.0, distance=2.0, with_detection=True)
    obs.detections[0].score = 0.2
    result = pipeline.process_frame(obs)
    assert result.objects == []


def test_duplicate_spawn_is_merged_back_into_original():
    pipeline = SemanticMappingPipeline(PipelineConfig(min_hits_to_confirm=1))
    obs = _observation(stamp=0.0, distance=2.0, with_detection=True)
    obs.detections.append(Detection2D(bbox=np.array([61.0, 41.0, 101.0, 81.0]), label="chair", score=0.9))
    result = pipeline.process_frame(obs)
    # Two detections of the same object in one frame: one spawns, the other
    # duplicate is merged into it (older ID kept) rather than kept as a second instance.
    assert len(result.objects) == 1
    assert result.objects[0].instance_id == 1


def test_detection_instance_ids_report_where_each_detection_went():
    pipeline = SemanticMappingPipeline(PipelineConfig(min_hits_to_confirm=1))
    first = pipeline.process_frame(_observation(stamp=0.0, distance=2.0, with_detection=True))
    assert first.detection_instance_ids == [1]  # spawned

    second = pipeline.process_frame(_observation(stamp=0.1, distance=2.0, with_detection=True))
    assert second.detection_instance_ids == [1]  # matched the same instance

    low = _observation(stamp=0.2, distance=2.0, with_detection=True)
    low.detections[0].score = 0.2
    low.detections.append(Detection2D(bbox=np.array([5.0, 5.0, 15.0, 15.0]), label="mug", score=0.2))
    third = pipeline.process_frame(low)
    assert third.detection_instance_ids == [1, -1]  # low-score: matched existing track / discarded


def test_depth_fill_recovers_points_and_change_detection_from_sparse_depth():
    rng = np.random.default_rng(0)
    keep = rng.random((120, 160)) < 0.05  # LiDAR-like: 5% of pixels carry a reading

    def sparse(stamp, distance, with_detection):
        obs = _observation(stamp, distance, with_detection)
        obs.depth = np.where(keep, obs.depth, 0.0)
        return obs

    def run(fill_radius):
        pipeline = SemanticMappingPipeline(PipelineConfig(min_hits_to_confirm=2, depth_fill_radius_px=fill_radius,
                                                          disappeared_occupied_fraction=0.3))
        for i in range(5):
            result = pipeline.process_frame(sparse(i * 0.1, 2.0, True))
        (obj,) = result.objects
        points_seen = obj.points_world.shape[0]
        for i in range(5, 45):
            result = pipeline.process_frame(sparse(i * 0.1, 8.0, False))  # object removed, surface behind it visible
        return points_seen, result.objects[0].status.value

    sparse_points, sparse_status = run(0)
    filled_points, filled_status = run(2)
    assert filled_points > 2 * sparse_points          # most of the silhouette back (voxel grid caps the count)
    assert filled_status == "disappeared"             # filled pixels carry free-space evidence for every point
    assert sparse_status == "disappeared"             # the sampled 5% still contradict enough points over 40 frames


def test_culling_leaves_the_map_identical_when_an_object_leaves_and_re_enters_the_view():
    looking_away = np.eye(4)
    looking_away[:3, :3] = np.diag([-1.0, 1.0, -1.0])  # camera turned 180 degrees: the object is behind it

    def frames():
        for i in range(5):
            yield _observation(i * 0.1, 2.0, True)
        for i in range(5, 15):
            obs = _observation(i * 0.1, 8.0, False)
            obs.pose = StampedPose(stamp=obs.stamp, T_world_from_frame=looking_away)
            yield obs
        for i in range(15, 20):
            yield _observation(i * 0.1, 2.0, True)

    final = {}
    for cull in (True, False):
        pipeline = SemanticMappingPipeline(PipelineConfig(min_hits_to_confirm=2, cull_out_of_view=cull))
        for obs in frames():
            result = pipeline.process_frame(obs)
        final[cull] = result.objects

    assert [o.instance_id for o in final[True]] == [o.instance_id for o in final[False]] == [1]
    a, b = final[True][0], final[False][0]
    assert (a.status, a.hits, a.frames_since_seen) == (b.status, b.hits, b.frames_since_seen) == (ObjectStatus.ACTIVE, 10, 0)
    np.testing.assert_array_equal(a.point_log_odds, b.point_log_odds)
    np.testing.assert_allclose(a.bbox3d, b.bbox3d)
