"""Regression tests for the paper discrepancies fixed from doc/paper-review-2026-09-25.md.

Each test names the review ID it pins down.
"""
import numpy as np

from semantic_mapping import association, tracking
from semantic_mapping import evaluation as ev
from semantic_mapping import scene_graph as sg
from semantic_mapping.geometry_utils import bbox3d_gap, iou_xyxy, overlap_3d
from semantic_mapping.pipeline import PipelineConfig, SemanticMappingPipeline
from semantic_mapping.types import Observation, ObjectStatus, StampedPose
from test.helpers import make_object
from semantic_mapping.vln import serialize_prompt as vp
from test.test_review_2026_09_24 import BOX, INTRINSICS, _depth, _det, _obs


# ------------------------------------------------------------------ D1
def test_d1_flickering_label_fuses_into_one_instance_and_eq10_moves_the_belief():
    pipeline = SemanticMappingPipeline()
    ids = set()
    for i, label in enumerate(["chair"] * 3 + ["armchair"] * 5):
        result = pipeline.process_frame(_obs(i * 0.1, [_det(label)]))
        ids.update(result.detection_instance_ids)
    (instance,) = pipeline.object_map.objects.values()
    assert ids == {instance.instance_id}
    assert instance.label == "armchair" and instance.label_belief["armchair"] > 0.9
    assert pipeline.object_map.stats["relabel_matches"] >= 1


def test_d1_one_wrong_label_only_dents_the_belief():
    pipeline = SemanticMappingPipeline()
    for i, label in enumerate(["chair"] * 3 + ["sofa"] + ["chair"] * 2):
        pipeline.process_frame(_obs(i * 0.1, [_det(label)]))
    (instance,) = pipeline.object_map.objects.values()
    assert instance.label == "chair" and instance.label_confidence > 0.99


def test_d1_relabelling_needs_3d_agreement_and_can_be_disabled():
    obj = make_object(1, "chair", [-0.2, -0.2, 1.9, 0.2, 0.2, 2.1])
    obj.points_world = np.zeros((10, 3))
    track = obj.track
    box = obj.track.state[:4]
    bbox2d = np.array([box[0] - box[2] / 2, box[1] - box[3] / 2, box[0] + box[2] / 2, box[1] + box[3] / 2])
    inside = np.array([-0.1, -0.1, 1.95, 0.1, 0.1, 2.05])   # a partial view of the chair
    in_front = np.array([-0.2, -0.2, 0.9, 0.2, 0.2, 1.3])   # a person 0.6 m in front of it
    for det_box, expected in ((inside, [(0, 0)]), (in_front, [])):
        result = association.associate_relabel([track], [bbox2d], [bbox2d], [det_box], [obj])
        assert result.matches == expected
    disabled = association.associate_relabel([track], [bbox2d], [bbox2d], [inside], [obj], min_overlap=0.0)
    assert disabled.matches == []
    assert overlap_3d(inside, obj.bbox3d) == 1.0 and overlap_3d(in_front, obj.bbox3d) == 0.0


def test_d1_relabelling_can_agree_with_the_visible_part_which_needs_no_motion_gate():
    obj = make_object(1, "chair", [-0.2, -0.2, 1.9, 0.2, 0.2, 2.1])
    obj.points_world = np.zeros((10, 3))
    track = tracking.init_track(BOX)
    inside = np.array([-0.1, -0.1, 1.95, 0.1, 0.1, 2.05])
    shifted = BOX + np.array([20.0, 0.0, 20.0, 0.0])  # IoU 1/3 with the prediction, outside the motion gate
    assert iou_xyxy(BOX, shifted) >= 0.3 and not tracking.mahalanobis_gate(track, shifted)

    def relabel(detection_box, visible):
        return association.associate_relabel([track], [BOX], [detection_box], [inside], [obj],
                                             visible_bboxes=visible).matches

    assert relabel(shifted, None) == []              # the prediction path keeps its motion gate
    assert relabel(shifted, [shifted]) == [(0, 0)]   # where the instance is seen now needs none
    quarter = np.array([60.0, 40.0, 70.0, 80.0])     # a detection covering a quarter of the box
    assert relabel(quarter, None) == [] and relabel(quarter, [BOX]) == []
    assert relabel(quarter, [quarter]) == [(0, 0)]


def _hidden_right_side(stamp, detections):
    """The chair at 2 m, its right three quarters behind something 1 m from the camera."""
    depth = _depth(2.0)
    depth[40:80, 70:100] = 1.0
    return Observation(stamp=stamp, pose=StampedPose(stamp=stamp, T_world_from_frame=np.eye(4)),
                       intrinsics=INTRINSICS, depth=depth, detections=list(detections))


def test_d1_a_new_label_on_the_visible_part_of_a_partly_hidden_object_is_a_relabelling():
    # One detection, under a label the chair has not taken, boxes only the
    # chair's left quarter (IoU 0.25 with the chair's box). With the rest of
    # the chair hidden, that quarter is all of it the camera sees: the same
    # object, for Eq. 10 to weigh. With the chair in full view, a box a
    # quarter its size is a smaller object on it (a book on a shelf) and
    # starts its own instance.
    quarter = [60.0, 40.0, 70.0, 80.0]
    for hidden, fused in ((True, True), (False, False)):
        pipeline = SemanticMappingPipeline()
        for i in range(3):
            pipeline.process_frame(_obs(i * 0.1, [_det()]))
        (chair_id,) = pipeline.object_map.objects
        detection = [_det("armchair", bbox=quarter)]
        result = pipeline.process_frame(_hidden_right_side(0.3, detection) if hidden else _obs(0.3, detection))
        assert (result.detection_instance_ids == [chair_id]) == fused
        assert ("armchair" in pipeline.object_map.objects[chair_id].label_belief) == fused
        assert len(pipeline.object_map.objects) == (1 if fused else 2)


# ------------------------------------------------------------------ D3
def test_d3_a_replacement_behind_the_same_image_box_is_a_disappearance_and_an_appearance():
    pipeline = SemanticMappingPipeline()
    for i in range(3):
        pipeline.process_frame(_obs(i * 0.1, [_det()], depth=2.0))
    (first,) = pipeline.object_map.objects.values()
    for i in range(3, 8):
        result = pipeline.process_frame(_obs(i * 0.1, [_det()], depth=6.0))
    assert result.detection_instance_ids[0] != first.instance_id
    assert first.status == ObjectStatus.DISAPPEARED and first.bbox3d[5] < 3.0, "its box never stretched to 6 m"
    assert pipeline.object_map.stats["depth_split_matches"] >= 1


def test_d3_validation_can_be_disabled():
    pipeline = SemanticMappingPipeline(PipelineConfig(match_max_gap_m=0.0))
    for i in range(3):
        pipeline.process_frame(_obs(i * 0.1, [_det()], depth=2.0))
    result = pipeline.process_frame(_obs(0.3, [_det()], depth=6.0))
    assert result.detection_instance_ids == [next(iter(pipeline.object_map.objects))]
    assert bbox3d_gap([0, 0, 0, 1, 1, 1], [1, 1, 3, 2, 2, 4]) == 2.0


# ------------------------------------------------------------------ D4
def test_d4_appeared_object_change_recall_differs_from_detection_recall():
    gt = ev.GroundTruthObject("chair", np.array([-0.5, -0.5, 1.5, 0.5, 0.5, 2.5]), appear_frame=2,
                              disappear_frame=10, visible_frames={2, 3})
    evaluator = ev.SequenceEvaluator([gt], INTRINSICS)
    for frame_id in range(4):
        evaluator.observe(frame_id, np.eye(4), [])  # never detected
    stats = evaluator.stats[0]
    # The DualMap artifact the paper points out: absence credit without detection.
    assert stats.detection_recall == 0.0 and stats.change_recall == 0.5


def test_d4_a_returning_object_scores_the_gap_between_its_phases_once():
    box = np.array([-0.5, -0.5, 1.5, 0.5, 0.5, 2.5])
    first = ev.GroundTruthObject("bag", box, 0, 2, {0, 1}, identity="bag#1")
    back = ev.GroundTruthObject("bag", box, 4, 6, {4, 5}, identity="bag#1")
    evaluator = ev.SequenceEvaluator([first, back], INTRINSICS)
    for frame_id in range(6):
        evaluator.observe(frame_id, np.eye(4), [])
    assert evaluator.stats[0].absence_frames == 2  # frames 2-3, after the first phase
    assert evaluator.stats[1].absence_frames == 0  # the same frames are not scored again


# ------------------------------------------------------------------ D21
def test_d21_a_relocation_on_appearance_alone_must_be_plausible():
    retired = make_object(1, "chair", [-0.3, -0.3, -0.3, 0.3, 0.3, 0.3], status=ObjectStatus.DISAPPEARED)
    retired.latest_stamp = 0.0
    retired.embedding = np.ones(8, dtype=np.float32) / np.sqrt(8)

    def match(offset, now, **bounds):
        box = np.array([-0.3, -0.3, -0.3, 0.3, 0.3, 0.3]) + np.array([offset, 0, 0] * 2)
        result, _ = association.reidentify([box], ["chair"], [retired.embedding], [retired], now=now, **bounds)
        return bool(result.matches)

    bounds = dict(relocation_max_distance=10.0, relocation_max_gap_sec=120.0)
    assert match(3.0, 30.0, **bounds)            # moved across the room shortly after: same object
    assert not match(20.0, 30.0, **bounds)       # too far to have been carried there
    assert not match(3.0, 300.0, **bounds)       # a lookalike turning up much later is a new object
    assert match(0.0, 1000.0, **bounds)          # back in its own place: geometry decides, no time bound
    assert match(20.0, 300.0)                    # 0 disables both bounds
    # A descriptor of another dimension (older map, other embedder) neither vetoes nor supports.
    other = [np.ones(5, dtype=np.float32)]
    same_place = np.array([-0.3, -0.3, -0.3, 0.3, 0.3, 0.3])
    assert association.reidentify([same_place], ["chair"], other, [retired], now=1.0)[0].matches == [(0, 0)]
    moved = same_place + np.array([3.0, 0, 0] * 2)
    assert association.reidentify([moved], ["chair"], other, [retired], now=1.0)[0].matches == []


# ------------------------------------------------------------------ D27
def _run(config, frames):
    pipeline = SemanticMappingPipeline(config)
    for i, (label, depth) in enumerate(frames):
        result = pipeline.process_frame(_obs(i * 0.1, [_det(label)] if label else [], depth=depth))
    return pipeline, result


def test_d27_without_geometric_consistency_a_removed_object_is_never_retired():
    frames = [("chair", 2.0)] * 3 + [(None, None)] * 6  # then the chair is gone: background at 8 m
    with_gc, _ = _run(PipelineConfig(), frames)
    without, _ = _run(PipelineConfig(use_geometric_consistency=False), frames)
    assert next(iter(with_gc.object_map.objects.values())).status == ObjectStatus.DISAPPEARED
    assert next(iter(without.object_map.objects.values())).status != ObjectStatus.DISAPPEARED


def test_d27_without_semantic_fusion_the_latest_label_wins():
    frames = [("chair", 2.0)] * 3 + [("armchair", 2.0)]
    fused, _ = _run(PipelineConfig(), frames)
    latest, _ = _run(PipelineConfig(use_semantic_fusion=False), frames)
    assert next(iter(fused.object_map.objects.values())).label == "chair"
    (instance,) = latest.object_map.objects.values()
    assert instance.label_belief == {"armchair": 1.0}


def test_d27_without_the_2d_tracker_association_runs_in_3d(monkeypatch):
    calls = []
    monkeypatch.setattr(association, "associate", lambda *a, **k: calls.append(1))
    pipeline, result = _run(PipelineConfig(use_2d_tracker=False), [("chair", 2.0)] * 4)
    assert not calls and len(pipeline.object_map.objects) == 1
    assert result.detection_instance_ids == [next(iter(pipeline.object_map.objects))]


# ------------------------------------------------------------------ D12
def test_d12_a_mug_on_a_long_table_is_on_it():
    table = make_object(1, "table", [0.0, 0.0, 0.0, 4.0, 1.0, 0.75])
    mug = make_object(2, "mug", [3.8, 0.4, 0.75, 3.9, 0.5, 0.85])  # IoU_xy 0.0025
    assert sg.SpatialEdge(2, "on", 1) in sg.build_spatial_edges([table, mug])
    assert sg.build_spatial_edges([table, mug], on_min_footprint_fraction=0.0) == []  # the paper's IoU test alone
    hanging = make_object(3, "mug", [3.95, 0.4, 0.75, 4.15, 0.5, 0.85])  # mostly past the table's end
    assert not any(e.predicate == "on" for e in sg.build_spatial_edges([table, hanging]))


# ------------------------------------------------------------------ D13, D14
def test_d14_a_disappeared_node_is_marked_on_its_line():
    plant = make_object(1, "plant", [0.0, 0.0, 0.0, 0.2, 0.2, 0.5], status=ObjectStatus.DISAPPEARED)
    text = vp.serialize_subgraph_to_text([plant], sg.SceneGraph(node_ids=[1]), include_temporal_cues=False)
    assert "Instance 1 (plant) at [0.10, 0.10, 0.25] (disappeared)" in text


def test_d13_a_moving_object_gets_a_timestamped_path():
    bag = make_object(4, "bag", [0.0, 0.0, 0.0, 0.3, 0.3, 0.3])
    bag.first_seen_stamp = 0.0
    bag.trajectory = [(t, np.array([x, 0.0, 0.15]), "active")
                      for t, x in ((2.0, 0.0), (3.0, 0.1), (5.0, 2.0), (9.0, 4.0), (12.0, 4.05))]
    text = vp.serialize_subgraph_to_text([bag], sg.SceneGraph(node_ids=[4]))
    assert "moved from [0.00, 0.00, 0.15] at t=2.00s" in text
    assert ("Instance 4 (bag) path: t=2.00s [0.00, 0.00, 0.15] -> t=5.00s [2.00, 0.00, 0.15] -> "
            "t=9.00s [4.00, 0.00, 0.15]") in text
    bag.trajectory = [(float(t), np.array([float(t), 0.0, 0.15]), "active") for t in range(2, 40)]
    path = next(line for line in vp.serialize_subgraph_to_text([bag], sg.SceneGraph(node_ids=[4])).splitlines()
                if "path:" in line)
    assert path.count("t=") == vp.MAX_PATH_POINTS and "t=2.00s" in path and "t=39.00s" in path


# ------------------------------------------------------------------ D16
def test_d16_offline_runs_can_detect_at_the_paper_rate():
    from semantic_mapping.datasets import run_sequence

    class Frames:
        def __iter__(self):
            for i in range(25):  # 10 Hz for 2.4 s
                yield type("Frame", (), {"frame_id": i, "stamp": i * 0.1, "rgb": None})()

        def observation(self, frame, detections):
            return _obs(frame.stamp, detections)

    class Detector:
        calls = []

        def detect(self, rgb, prompts=None, frame_id=None):
            self.calls.append(frame_id)
            return [_det()]

    detector = Detector()
    evaluated = [bool(r.detection_instance_ids) for _f, _d, r in
                 run_sequence(Frames(), SemanticMappingPipeline(), detector, None, detector_rate_hz=1.0)]
    assert detector.calls == [0, 10, 20] and sum(evaluated) == 3
    detector.calls.clear()
    list(run_sequence(Frames(), SemanticMappingPipeline(), detector, None))
    assert len(detector.calls) == 25
