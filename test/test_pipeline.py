import numpy as np

from semantic_mapping.pipeline import PipelineConfig, SemanticMappingPipeline
from semantic_mapping.types import CameraIntrinsics, Detection2D, Observation, StampedPose

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
