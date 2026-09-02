import numpy as np

from semantic_mapping import evaluation as ev
from semantic_mapping.types import CameraIntrinsics, ObjectStatus
from test.helpers import make_object

INTRINSICS = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=40.0, width=100, height=80)
CAMERA_AT_ORIGIN = np.eye(4)  # optical frame: looks down +z


def _gt(label="chair", bbox=(-0.5, -0.5, 1.5, 0.5, 0.5, 2.5), appear=0, disappear=10, visible=range(10)):
    return ev.GroundTruthObject(
        label=label, bbox3d=np.array(bbox, dtype=np.float64),
        appear_frame=appear, disappear_frame=disappear, visible_frames=set(visible),
    )


def _instance(instance_id, label, bbox, status=ObjectStatus.ACTIVE):
    obj = make_object(instance_id, label, bbox, status=status)
    obj.points_world = np.array([[bbox[0], bbox[1], bbox[2]]])  # non-empty so is_match considers it
    return obj


def test_is_match_requires_label_iou_and_centroid():
    gt = _gt()
    assert ev.is_match(_instance(1, "chair", (-0.5, -0.5, 1.5, 0.5, 0.5, 2.5)), gt)
    assert not ev.is_match(_instance(1, "table", (-0.5, -0.5, 1.5, 0.5, 0.5, 2.5)), gt)
    assert not ev.is_match(_instance(1, "chair", (5.0, 5.0, 1.5, 6.0, 6.0, 2.5)), gt)  # no overlap
    # Overlaps but centroid too far: a long box sharing one corner region.
    assert not ev.is_match(_instance(1, "chair", (0.3, 0.3, 1.5, 2.0, 2.0, 2.5)), gt)


def test_location_in_view():
    in_front = _gt(bbox=(-0.1, -0.1, 1.9, 0.1, 0.1, 2.1))
    behind = _gt(bbox=(-0.1, -0.1, -2.1, 0.1, 0.1, -1.9))
    assert ev.location_in_view(in_front, INTRINSICS, CAMERA_AT_ORIGIN)
    assert not ev.location_in_view(behind, INTRINSICS, CAMERA_AT_ORIGIN)


def test_detection_recall_and_fragments():
    gt = _gt(appear=0, disappear=4, visible=[0, 1, 2, 3])
    evaluator = ev.SequenceEvaluator([gt], INTRINSICS)
    box = (-0.5, -0.5, 1.5, 0.5, 0.5, 2.5)
    evaluator.observe(0, CAMERA_AT_ORIGIN, [_instance(1, "chair", box)])
    evaluator.observe(1, CAMERA_AT_ORIGIN, [_instance(1, "chair", box)])
    evaluator.observe(2, CAMERA_AT_ORIGIN, [])                              # lost it
    evaluator.observe(3, CAMERA_AT_ORIGIN, [_instance(7, "chair", box)])     # re-spawned with a new ID
    stats = evaluator.stats[0]
    assert stats.appearance_frames == 4
    assert stats.detection_recall == 0.75
    assert stats.fragments == 2


def test_invisible_frames_are_excluded_from_appearance_interval():
    gt = _gt(appear=0, disappear=4, visible=[0, 1])  # present for 4 frames, only visible for 2
    evaluator = ev.SequenceEvaluator([gt], INTRINSICS)
    for frame_id in range(4):
        evaluator.observe(frame_id, CAMERA_AT_ORIGIN, [])
    assert evaluator.stats[0].appearance_frames == 2


def test_change_recall_penalizes_stale_active_instance_after_removal():
    gt = _gt(appear=0, disappear=2, visible=[0, 1])  # removed at frame 2, location stays in view
    evaluator = ev.SequenceEvaluator([gt], INTRINSICS)
    box = (-0.5, -0.5, 1.5, 0.5, 0.5, 2.5)
    evaluator.observe(0, CAMERA_AT_ORIGIN, [_instance(1, "chair", box)])
    evaluator.observe(1, CAMERA_AT_ORIGIN, [_instance(1, "chair", box)])
    evaluator.observe(2, CAMERA_AT_ORIGIN, [_instance(1, "chair", box)])                              # stale: FP
    evaluator.observe(3, CAMERA_AT_ORIGIN, [_instance(1, "chair", box, status=ObjectStatus.DISAPPEARED)])  # confirmed
    stats = evaluator.stats[0]
    assert stats.absence_frames == 2
    assert stats.absence_hits == 1
    assert stats.change_recall == 0.75  # (2 + 1) / (2 + 2)


def test_never_present_object_gets_no_absence_credit():
    gt = _gt(appear=5, disappear=10, visible=[])  # appears late, never observed
    evaluator = ev.SequenceEvaluator([gt], INTRINSICS)
    for frame_id in range(5):
        evaluator.observe(frame_id, CAMERA_AT_ORIGIN, [])
    assert evaluator.stats[0].absence_frames == 0


def test_final_map_prf_counts_duplicates_as_false_positives():
    gt = _gt(appear=0, disappear=10)
    evaluator = ev.SequenceEvaluator([gt], INTRINSICS)
    box = (-0.5, -0.5, 1.5, 0.5, 0.5, 2.5)
    evaluator.observe(0, CAMERA_AT_ORIGIN, [_instance(1, "chair", box), _instance(2, "chair", box)])
    precision, recall, f1, tp, n_pred, n_gt = evaluator.final_map_prf()
    assert (tp, n_pred, n_gt) == (1, 2, 1)
    assert precision == 0.5 and recall == 1.0
    assert abs(f1 - 2 / 3) < 1e-9


def test_summary_and_format_run():
    gt = _gt(appear=0, disappear=2, visible=[0, 1])
    evaluator = ev.SequenceEvaluator([gt], INTRINSICS)
    evaluator.observe(0, CAMERA_AT_ORIGIN, [_instance(1, "chair", (-0.5, -0.5, 1.5, 0.5, 0.5, 2.5))])
    evaluator.observe(1, CAMERA_AT_ORIGIN, [_instance(1, "chair", (-0.5, -0.5, 1.5, 0.5, 0.5, 2.5))])
    summary = evaluator.summary()
    assert summary["per_object"][0]["fragments"] == 1
    assert "mean detection recall" in ev.format_summary(summary)


def test_location_in_view_respects_occlusion_from_depth_image():
    gt = _gt(bbox=(-0.1, -0.1, 1.9, 0.1, 0.1, 2.1))  # centroid projects to (50, 40) at z = 2
    clear = np.full((80, 100), 6.0)      # sensor sees the wall behind the spot: observable
    blocked = np.full((80, 100), 1.0)    # something 1 m in front: occluded
    assert ev.location_in_view(gt, INTRINSICS, CAMERA_AT_ORIGIN, clear)
    assert not ev.location_in_view(gt, INTRINSICS, CAMERA_AT_ORIGIN, blocked)
