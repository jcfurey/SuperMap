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
