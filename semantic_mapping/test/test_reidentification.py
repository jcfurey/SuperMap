"""Identity persistence across disappearance and relocation (stage-4 association)."""
import numpy as np

from semantic_mapping import association
from semantic_mapping.pipeline import PipelineConfig, SemanticMappingPipeline
from semantic_mapping.types import CameraIntrinsics, Detection2D, ObjectStatus, Observation, StampedPose
from semantic_mapping.vln import serialize_prompt as vp
from semantic_mapping import scene_graph as sg
from test.helpers import make_object

INTRINSICS = CameraIntrinsics(fx=100.0, fy=100.0, cx=80.0, cy=60.0, width=160, height=120)
LEFT, RIGHT = (30, 70), (95, 135)      # two column ranges for a 40 px wide object


def _retired(instance_id, label, bbox, embedding=None, latest_stamp=0.0):
    obj = make_object(instance_id, label, bbox, status=ObjectStatus.DISAPPEARED)
    obj.embedding = None if embedding is None else np.asarray(embedding, dtype=np.float32)
    obj.embedding_count = 0 if embedding is None else 3
    obj.latest_stamp = latest_stamp
    return obj


def test_reidentify_by_place_and_by_appearance():
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    blue = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    chair = _retired(7, "chair", [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], red)
    boxes = [np.array([0.1, 0.1, 0.0, 0.9, 0.9, 1.0]), np.array([5.0, 5.0, 0.0, 6.0, 6.0, 1.0])]

    # Same place, same look: matched by place.
    result, by_place = association.reidentify(boxes[:1], ["chair"], [red], [chair])
    assert result.matches == [(0, 0)] and by_place == {(0, 0)}
    # Same place, different look: a different chair was put there.
    result, _ = association.reidentify(boxes[:1], ["chair"], [blue], [chair])
    assert result.matches == []
    # Elsewhere, same look: relocation.
    result, by_place = association.reidentify(boxes[1:], ["chair"], [red], [chair])
    assert result.matches == [(0, 0)] and by_place == set()
    # Elsewhere, no appearance on either side: nothing to claim a relocation with.
    plain = _retired(8, "chair", [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    result, _ = association.reidentify(boxes[1:], ["chair"], [None], [plain])
    assert result.matches == []
    # Same place with no appearance: label + geometry suffice.
    result, by_place = association.reidentify(boxes[:1], ["chair"], [None], [plain])
    assert result.matches == [(0, 0)] and by_place == {(0, 0)}
    # Wrong label never matches; too old never matches.
    assert association.reidentify(boxes[:1], ["table"], [red], [chair])[0].matches == []
    assert association.reidentify(boxes[:1], ["chair"], [red], [chair], max_age_sec=5.0, now=100.0)[0].matches == []


def _observation(stamp, columns=None, colour=(200, 40, 40), distance=2.0, occluder=None):
    """A coloured 40x40 px plane at ``distance`` in the given column range (None: empty room);
    ``occluder`` puts an undetected grey surface 1 m in front of that column range."""
    depth = np.full((120, 160), 8.0)
    rgb = np.full((120, 160, 3), 235, dtype=np.uint8)
    detections = []
    if occluder is not None:
        o1, o2 = occluder
        depth[30:90, o1 - 5:o2 + 5] = 1.0
        rgb[30:90, o1 - 5:o2 + 5] = (120, 120, 120)
    if columns is not None:
        c1, c2 = columns
        depth[40:80, c1:c2] = distance
        rgb[40:80, c1:c2] = colour
        mask = np.zeros((120, 160), dtype=bool)
        mask[40:80, c1:c2] = True
        detections = [Detection2D(bbox=np.array([c1, 40.0, c2, 80.0], dtype=np.float64), label="box", score=0.9, mask=mask)]
    return Observation(stamp=stamp, pose=StampedPose(stamp=stamp, T_world_from_frame=np.eye(4)),
                       intrinsics=INTRINSICS, rgb=rgb, depth=depth, detections=detections)


def _run(pipeline, frames):
    result = None
    for obs in frames:
        result = pipeline.process_frame(obs)
    return result


def test_object_that_returns_to_its_place_keeps_its_id():
    pipeline = SemanticMappingPipeline(PipelineConfig(min_hits_to_confirm=2, disappeared_occupied_fraction=0.3))
    present = [_observation(i * 0.1, LEFT) for i in range(5)]
    absent = [_observation(i * 0.1) for i in range(5, 45)]
    result = _run(pipeline, present + absent)
    (box,) = result.objects
    assert box.status == ObjectStatus.DISAPPEARED
    original_id, hits_before = box.instance_id, box.hits

    result = _run(pipeline, [_observation(i * 0.1, LEFT) for i in range(45, 50)])
    assert [o.instance_id for o in result.objects] == [original_id]
    assert result.objects[0].status == ObjectStatus.ACTIVE and result.objects[0].hits > hits_before
    assert result.objects[0].embedding is not None and result.objects[0].embedding_count >= 10

    text = vp.serialize_subgraph_to_text(result.objects, sg.build_scene_graph(result.objects))
    assert "reappeared" in text and "same place" in text


def test_relocated_object_keeps_its_id_through_appearance_and_records_the_move():
    pipeline = SemanticMappingPipeline(PipelineConfig(min_hits_to_confirm=2, disappeared_occupied_fraction=0.3))
    present = [_observation(i * 0.1, LEFT) for i in range(5)]
    absent = [_observation(i * 0.1) for i in range(5, 45)]
    _run(pipeline, present + absent)
    (box,) = pipeline.object_map.objects.values()
    original_id = box.instance_id
    old_center = box.center.copy()

    result = _run(pipeline, [_observation(i * 0.1, RIGHT) for i in range(45, 50)])
    assert [o.instance_id for o in result.objects] == [original_id]
    box = result.objects[0]
    assert box.status == ObjectStatus.ACTIVE
    assert np.linalg.norm(box.center - old_center) > 1.0          # the geometry now describes the new place
    assert box.bbox3d[0] > 0.0 > old_center[0] - 0.5                  # left of the axis before, right of it now

    text = vp.serialize_subgraph_to_text(result.objects, sg.build_scene_graph(result.objects))
    assert "reappeared" in text and "moved from" in text


def test_a_different_looking_object_in_the_old_place_gets_a_new_id():
    pipeline = SemanticMappingPipeline(PipelineConfig(min_hits_to_confirm=2, disappeared_occupied_fraction=0.3))
    _run(pipeline, [_observation(i * 0.1, LEFT) for i in range(5)] + [_observation(i * 0.1) for i in range(5, 45)])
    (red_box,) = pipeline.object_map.objects.values()

    result = _run(pipeline, [_observation(i * 0.1, LEFT, colour=(40, 40, 200)) for i in range(45, 50)])
    ids = sorted(o.instance_id for o in result.objects)
    assert ids == [red_box.instance_id, red_box.instance_id + 1]
    statuses = {o.instance_id: o.status for o in result.objects}
    assert statuses[red_box.instance_id] == ObjectStatus.DISAPPEARED
    assert statuses[red_box.instance_id + 1] == ObjectStatus.ACTIVE


def test_reid_can_be_disabled():
    pipeline = SemanticMappingPipeline(PipelineConfig(min_hits_to_confirm=2, disappeared_occupied_fraction=0.3,
                                                      reid_enabled=False))
    _run(pipeline, [_observation(i * 0.1, LEFT) for i in range(5)] + [_observation(i * 0.1) for i in range(5, 45)])
    result = _run(pipeline, [_observation(i * 0.1, LEFT) for i in range(45, 50)])
    assert len(result.objects) == 2


def test_object_moved_before_its_old_spot_is_confirmed_empty_is_reconciled_under_its_original_id():
    pipeline = SemanticMappingPipeline(PipelineConfig(min_hits_to_confirm=2, disappeared_occupied_fraction=0.3))
    _run(pipeline, [_observation(i * 0.1, LEFT) for i in range(5)])
    (box,) = pipeline.object_map.objects.values()
    original_id, old_center = box.instance_id, box.center.copy()

    # Something stands in front of the old spot while the box shows up on the
    # right: the old instance is still live, so a provisional instance appears.
    result = _run(pipeline, [_observation(i * 0.1, RIGHT, occluder=LEFT) for i in range(5, 12)])
    ids = sorted(o.instance_id for o in result.objects)
    assert ids == [original_id, original_id + 1]
    assert pipeline.object_map.objects[original_id].status != ObjectStatus.DISAPPEARED

    # The old spot comes into view and is empty: the old instance retires and
    # the provisional record folds into the original ID.
    result = _run(pipeline, [_observation(i * 0.1, RIGHT) for i in range(12, 40)])
    assert [o.instance_id for o in result.objects] == [original_id]
    box = result.objects[0]
    assert box.status == ObjectStatus.ACTIVE and np.linalg.norm(box.center - old_center) > 1.0
    statuses = [s[2] for s in box.trajectory]
    assert "disappeared" in statuses and statuses[-1] == "active"
    text = vp.serialize_subgraph_to_text(result.objects, sg.build_scene_graph(result.objects))
    assert "moved from" in text
