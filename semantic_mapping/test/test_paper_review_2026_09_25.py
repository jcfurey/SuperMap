"""Regression tests for the paper discrepancies fixed from doc/paper-review-2026-09-25.md.

Each test names the review ID it pins down.
"""
import numpy as np

from semantic_mapping import association
from semantic_mapping import evaluation as ev
from semantic_mapping.geometry_utils import bbox3d_gap, overlap_3d
from semantic_mapping.pipeline import PipelineConfig, SemanticMappingPipeline
from semantic_mapping.types import ObjectStatus
from test.helpers import make_object
from test.test_review_2026_09_24 import INTRINSICS, _det, _obs


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
