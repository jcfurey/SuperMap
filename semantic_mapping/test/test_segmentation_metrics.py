import numpy as np
import pytest

from semantic_mapping import segmentation_metrics as sm
from semantic_mapping.types import ObjectStatus
from test.helpers import make_object


def _patch(x0: float, y0: float, z: float = 0.5, n: int = 6, step: float = 0.1) -> np.ndarray:
    xs, ys = np.meshgrid(x0 + step * np.arange(n), y0 + step * np.arange(n))
    return np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)], axis=1)


def _scene() -> tuple[sm.GroundTruthPoints, dict[str, np.ndarray]]:
    parts = {
        "chair0": _patch(0.0, 0.0), "chair1": _patch(3.0, 0.0), "table": _patch(6.0, 0.0),
        "floor": _patch(0.0, 3.0, z=0.0, n=10),
    }
    labels = ["chair"] * 36 + ["chair"] * 36 + ["table"] * 36 + ["floor"] * 100
    instance_ids = [0] * 36 + [1] * 36 + [2] * 36 + [-1] * 100
    gt = sm.GroundTruthPoints(np.concatenate(list(parts.values())), labels, instance_ids)
    return gt, parts


def _instance(instance_id: int, label: str, points: np.ndarray, confidence: float = 0.9,
              status=ObjectStatus.ACTIVE):
    obj = make_object(instance_id, label, [0, 0, 0, 1, 1, 1], status=status)
    obj.points_world = np.asarray(points, dtype=np.float64)
    obj.label_belief = {label: confidence}
    return obj


def test_perfect_map_scores_one_without_background_and_penalizes_missing_background():
    gt, parts = _scene()
    objects = [_instance(1, "chair", parts["chair0"]), _instance(2, "chair", parts["chair1"]),
               _instance(3, "table", parts["table"])]
    report = sm.segmentation_report(objects, gt)

    without = report["class_level"]["without_background"]
    assert without["miou"] == pytest.approx(1.0) and without["fmiou"] == pytest.approx(1.0)
    assert without["acc"] == pytest.approx(1.0) and without["num_points"] == 108

    with_bg = report["class_level"]["with_background"]
    assert with_bg["classes"]["floor"]["iou"] == 0.0
    assert with_bg["miou"] == pytest.approx(2 / 3)
    assert with_bg["acc"] == pytest.approx(108 / 208)

    inst = report["instance_level"]
    assert inst["ap25"]["map"] == pytest.approx(1.0) and inst["ap50"]["map"] == pytest.approx(1.0)
    assert inst["num_gt_instances"] == 3 and inst["num_predictions"] == 3
    assert report["num_transferred_points"] == 108


def test_missed_instance_and_wrong_label():
    gt, parts = _scene()
    objects = [_instance(1, "chair", parts["chair0"]), _instance(3, "sofa", parts["table"])]
    report = sm.segmentation_report(objects, gt)

    classes = report["class_level"]["without_background"]["classes"]
    assert classes["chair"]["iou"] == pytest.approx(0.5)          # one of two chairs found
    assert classes["table"]["iou"] == 0.0 and classes["table"]["fn"] == 36
    assert "sofa" not in classes                                  # not a ground-truth class here
    assert report["class_level"]["without_background"]["acc"] == pytest.approx(36 / 108)

    ap50 = report["instance_level"]["ap50"]["per_class"]
    assert ap50["chair"] == pytest.approx(0.5)
    assert ap50["table"] == 0.0


def test_label_transfer_respects_radius_and_reports_unsupported_volume():
    gt, parts = _scene()
    far = _instance(1, "chair", parts["chair0"] + np.array([0.0, 0.0, 0.5]))
    transfer = sm.transfer_labels([far], gt, max_distance=0.1)
    assert np.all(transfer.pred_instance_ids == -1)
    assert transfer.unsupported_points[1] == 36

    near = _instance(2, "chair", parts["chair0"] + np.array([0.0, 0.0, 0.05]))
    transfer = sm.transfer_labels([near], gt, max_distance=0.1)
    assert np.sum(transfer.pred_instance_ids == 2) == 36
    assert transfer.unsupported_points[2] == 0


def test_disappeared_instances_are_not_predictions():
    gt, parts = _scene()
    gone = _instance(1, "chair", parts["chair0"], status=ObjectStatus.DISAPPEARED)
    transfer = sm.transfer_labels([gone], gt)
    assert np.all(transfer.pred_instance_ids == -1) and 1 not in transfer.instances


def test_partial_overlap_passes_low_threshold_only():
    gt, parts = _scene()
    quarter = _instance(1, "chair", parts["chair0"][:14])  # 14 / 36 = 0.39 IoU with chair0
    report = sm.segmentation_report([quarter], gt)
    per_class25 = report["instance_level"]["ap25"]["per_class"]
    per_class50 = report["instance_level"]["ap50"]["per_class"]
    assert per_class25["chair"] == pytest.approx(0.5)
    assert per_class50["chair"] == 0.0


def test_average_precision_uses_precision_envelope():
    ap = sm._average_precision(np.array([True, False, True]), n_gt=2)
    assert ap == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3))
    assert sm._average_precision(np.zeros(0, dtype=bool), n_gt=2) == 0.0
    assert np.isnan(sm._average_precision(np.array([True]), n_gt=0))


def test_aliases_fold_vocabulary_onto_ground_truth_names():
    gt, parts = _scene()
    objects = [_instance(1, "Couch", parts["chair0"])]
    gt.labels[:36] = "sofa"
    report = sm.segmentation_report(objects, gt, aliases={"couch": "sofa"})
    assert report["class_level"]["without_background"]["classes"]["sofa"]["iou"] == pytest.approx(1.0)


def test_ground_truth_points_roundtrip(tmp_path):
    gt, _ = _scene()
    gt.save(tmp_path / "gt_points.npz")
    loaded = sm.GroundTruthPoints.load(tmp_path / "gt_points.npz")
    assert np.allclose(loaded.points, gt.points, atol=1e-6)
    assert list(loaded.labels) == list(gt.labels) and np.array_equal(loaded.instance_ids, gt.instance_ids)


def test_format_report_mentions_every_table():
    gt, parts = _scene()
    text = sm.format_segmentation_report(sm.segmentation_report([_instance(1, "chair", parts["chair0"])], gt))
    assert "without background" in text and "AP25" in text and "mAP" in text
