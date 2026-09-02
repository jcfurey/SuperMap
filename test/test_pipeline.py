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
