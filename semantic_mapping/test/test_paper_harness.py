import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from semantic_mapping import evaluation, paper_harness as harness
from semantic_mapping import segmentation_metrics as seg
from semantic_mapping.datasets import load_dataset, load_prompts, load_yaml_params
from semantic_mapping.detectors import build_detector
from test.helpers import make_object

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reference():
    return harness.load_paper_results()


@pytest.fixture(scope="module")
def change_scene(tmp_path_factory):
    """The synthetic change scene, generated for this module."""
    spec = importlib.util.spec_from_file_location("prepare_example_dataset", ROOT / "examples/prepare_example_dataset.py")
    module = sys.modules.setdefault(spec.name, importlib.util.module_from_spec(spec))  # dataclasses look it up
    spec.loader.exec_module(module)
    return module.generate_scene(tmp_path_factory.mktemp("paper") / "change_scene", verbose=False)


def test_paper_results_are_internally_consistent(reference):
    for name, row in reference["table5"]["configurations"].items():
        p, r = row["precision"], row["recall"]
        assert row["f1"] == pytest.approx(2 * p * r / (p + r), abs=2e-4), name  # the paper rounds to 4 places
    table4 = reference["table4"]
    for method in table4["methods"].values():
        assert set(method) == set(table4["appeared"] + table4["disappeared"])
        assert all(0.0 <= v <= 1.0 for row in method.values() for v in row.values())
    for method in reference["table2"]["methods"].values():
        assert all(0.0 <= v <= 100.0 for setting in ("without_background", "with_background")
                   for v in method[setting].values())
    for method in reference["table3"]["methods"].values():
        assert set(method) == set(reference["table3"]["classes"])
    assert reference["protocol"]["tp_iou_3d"] == 0.1 and reference["protocol"]["tp_centroid_m"] == 0.3


def test_verdicts_and_what_fails():
    assert harness.compare(0.5, 0.52, "reproduction", 0.05) == harness.REPRODUCED
    assert harness.compare(0.5, 0.6, "reproduction", 0.05) == harness.ABOVE
    assert harness.compare(0.5, 0.4, "reproduction", 0.05) == harness.BELOW
    assert harness.compare(0.5, 0.5, "floor") == harness.AT_OR_ABOVE
    assert harness.compare(0.5, 0.49, "floor") == harness.BELOW
    assert harness.compare(0.5, float("nan"), "floor") == harness.NOT_RUN

    floor = harness.Check("a", "T", "floor", "d", 0.5, 0.4, harness.BELOW, "data")
    repro = harness.Check("b", "T", "reproduction", "d", 0.5, 0.4, harness.BELOW, "data")
    claim = harness.Check("c", "T", "claim", "d", "holds", "holds", harness.HOLDS, "data")
    assert floor.failed() and not repro.failed() and repro.failed(strict=True) and not claim.failed()

    stale = harness.apply_known_deviations([floor, repro, claim], {"a": "why", "c": "was fixed", "zzz": "gone"})
    assert not floor.failed() and floor.missed() and floor.known_deviation == "why"
    assert stale == ["c"]  # listed but no longer missing; unknown ids are ignored
    report = harness.format_report([floor, repro, claim])
    assert "below paper (known)" in report and "known deviation: why" in report and "None failed." in report


def test_known_deviations_name_real_checks(reference, change_scene):
    deviations = harness.load_known_deviations()
    dataset = load_dataset(change_scene)
    run = harness.run_sequence(dataset, build_detector("offline", detections_dir=dataset.detections_dir),
                               load_prompts(ROOT / "config/prompts.yaml"),
                               load_yaml_params(ROOT / "config/semantic_mapping.yaml"),
                               evaluation.load_ground_truth(change_scene / "scene_ground_truth.json")[1])
    checks = harness.table4_checks(run.evaluator, reference, "synthetic")
    checks += harness.identity_checks(run.evaluator, reference, "synthetic")
    ids = {check.id for check in checks}
    assert deviations and set(deviations) <= ids
    assert not harness.apply_known_deviations(checks, deviations)  # every entry still describes a real miss
    assert not [check.id for check in checks if check.failed()]
    by_id = {check.id: check for check in checks}
    assert by_id["table4.cart.detection"].verdict == harness.AT_OR_ABOVE
    assert by_id["identity.bucket.new"].verdict == harness.HOLDS
    assert by_id["identity.static.kept"].verdict == harness.HOLDS
    assert by_id["table4.beats_dualmap"].verdict == harness.NOT_RUN
    assert run.timings_ms["total"] > 0 and harness.runtime_checks(run.timings_ms, reference, "")[0].measured > 0


def test_table5_claim_needs_higher_f1_and_no_lower_precision_or_recall(reference):
    def rows(full, **ablations):
        base = {name: {"precision": 0.5, "recall": 0.5, "f1": 0.5} for name in harness.TABLE5_ABLATIONS}
        base.update(ablations)
        return {**base, harness.TABLE5_FULL: full}

    better = {"precision": 0.8, "recall": 0.6, "f1": 0.69}
    checks = {c.id: c for c in harness.table5_checks([rows(better)], reference, "d")}
    assert all(checks[f"table5.outperforms.{n}"].verdict == harness.HOLDS for n in harness.TABLE5_ABLATIONS)
    assert checks["table5.All (proposed).f1"].verdict == harness.NOT_RUN  # values are not compared off the paper's data

    tied = rows({"precision": 0.5, "recall": 0.5, "f1": 0.5})
    lower_recall = rows(better, **{"W/o Semantic Fusion": {"precision": 0.4, "recall": 0.7, "f1": 0.52}})
    checks = {c.id: c for c in harness.table5_checks([tied, lower_recall], reference, "d")}
    fusion = checks["table5.outperforms.W/o Semantic Fusion"]
    assert fusion.verdict == harness.FAILS and "lower recall" in fusion.detail and "1/2 runs" in fusion.detail

    paper = reference["table5"]["configurations"]
    reproduced = {c.id: c for c in harness.table5_checks([paper], reference, "d", reproduction=True)}
    assert all(c.verdict in (harness.REPRODUCED, harness.HOLDS) for c in reproduced.values())


def _segmentation_report(acc, ap50):
    per_class = {"chair": ap50, "window": ap50, "refridgerator": ap50, "sofa": ap50, "door": ap50}
    level = {"miou": acc, "fmiou": acc, "acc": acc}
    return {"class_level": {"without_background": level, "with_background": level},
            "instance_level": {"ap50": {"per_class": per_class}, "ap25": {"per_class": dict(per_class)}}}


def test_scannet_tables_compare_values_and_the_papers_claims(reference):
    aliases = {"refrigerator": "refridgerator"}
    good = _segmentation_report(0.5548, 0.9)
    table2 = {c.id: c for c in harness.table2_checks(good, reference, "scannet")}
    assert table2["table2.without_background.acc"].verdict == harness.REPRODUCED
    assert table2["table2.acc_beats_object_baselines"].verdict == harness.HOLDS
    table3 = {c.id: c for c in harness.table3_checks(good, reference, "scannet", aliases)}
    assert table3["table3.refrigerator.ap50"].measured == pytest.approx(90.0)  # found under ScanNet's spelling
    assert table3["table3.beats_baselines"].verdict == harness.HOLDS

    poor = _segmentation_report(0.30, 0.01)
    table2 = {c.id: c for c in harness.table2_checks(poor, reference, "scannet")}
    assert table2["table2.acc_beats_object_baselines"].verdict == harness.FAILS
    table3 = {c.id: c for c in harness.table3_checks(poor, reference, "scannet", aliases)}
    assert table3["table3.beats_baselines"].verdict == harness.FAILS  # 1 % < HOV-SG's 4.58 % on chair
    assert all(c.verdict == harness.NOT_RUN for c in harness.table2_checks(None, reference, ""))


def test_pooled_instance_ap_ranks_all_scenes_together():
    def scene(points, labels, ids, instances):
        gt = seg.GroundTruthPoints(points, labels, ids)
        objects = []
        for instance_id, (confidence, members) in instances.items():
            obj = make_object(instance_id, "chair", [0, 0, 0, 1, 1, 1])
            obj.points_world = points[members]
            obj.point_log_odds = np.zeros(len(members))
            obj.label_belief = {"chair": confidence}
            objects.append(obj)
        return objects, gt

    a = np.array([[0.0, 0, 0], [0.1, 0, 0]])
    b = np.array([[5.0, 0, 0], [5.1, 0, 0], [9.0, 0, 0]])
    first = scene(a, ["chair", "chair"], [0, 0], {1: (0.9, [0, 1])})                        # true positive
    second = scene(b, ["chair", "chair", "wall"], [0, 0, 1], {1: (0.8, [2]), 2: (0.7, [0, 1])})  # FP, then TP
    report = seg.pooled_segmentation_report([first, second], max_distance=0.05, instance_classes=["chair"])
    # Ranked 0.9 TP, 0.8 FP, 0.7 TP over 2 ground-truth chairs: AP = 0.5 * 1 + 0.5 * 2/3.
    assert report["instance_level"]["ap50"]["per_class"]["chair"] == pytest.approx(0.5 + 0.5 * 2 / 3)
    single = seg.pooled_segmentation_report([first], max_distance=0.05, instance_classes=["chair"])
    assert single["instance_level"] == seg.segmentation_report(
        *first, max_distance=0.05, instance_classes=["chair"])["instance_level"]
    assert report["class_level"]["with_background"]["num_points"] == 5


def test_cached_detections_replay_exactly(tmp_path):
    from semantic_mapping.detectors.offline import OfflineDetector, write_detections
    from semantic_mapping.types import Detection2D

    mask = np.zeros((6, 8), dtype=bool)
    mask[1:4, 2:5] = True
    detections = [Detection2D(np.array([2.0, 1.0, 5.0, 4.0]), "chair", 0.75, mask=mask),
                  Detection2D(np.array([0.5, 0.5, 3.0, 2.0]), "door", 0.5)]
    write_detections(tmp_path / "detections", 7, detections)
    replayed = OfflineDetector(tmp_path / "detections").detect(np.zeros((6, 8, 3)), frame_id=7)
    assert [(d.label, d.score, d.bbox.tolist()) for d in replayed] == \
        [(d.label, d.score, d.bbox.tolist()) for d in detections]
    assert np.array_equal(replayed[0].mask, mask) and replayed[1].mask is None


def test_cli_scannet_and_capture_modes(tmp_path, change_scene):
    """The ScanNet path on a tiny annotated scene and the capture path on the synthetic scene."""
    pytest.importorskip("cv2")
    from test.test_datasets_scannet import _write_scene

    scannet = _write_scene(tmp_path)
    (scannet / "detections").mkdir()
    for frame_id in (0, 10):
        (scannet / "detections" / f"{frame_id:06d}.json").write_text(json.dumps(
            {"detections": [{"bbox": [2, 2, 14, 10], "label": "chair", "score": 0.9}]}))
    out = tmp_path / "report.json"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples/paper_harness.py"), "--skip_synthetic", "--frame_skip", "1",
         "--scannet", str(scannet), "--capture", str(change_scene), "--json", str(out),
         "--markdown", str(tmp_path / "report.md")],
        capture_output=True, text=True, cwd=tmp_path)
    assert completed.returncode in (0, 1), completed.stderr
    checks = {row["id"]: row for row in json.loads(out.read_text())}
    assert checks["table2.without_background.acc"]["verdict"] != harness.NOT_RUN
    assert "ScanNet, 1 scene " in checks["table2.without_background.acc"]["data"]
    assert checks["table3.window.ap50"]["verdict"] == harness.NOT_RUN  # no window in the scene
    assert checks["table4.cart.detection"]["kind"] == "floor"
    assert checks["table5.outperforms.W/o 2D Tracker"]["kind"] == "claim"
    assert "| Table IV |" in (tmp_path / "report.md").read_text()
