#!/usr/bin/env python3
"""Check this package against the results SuperMap's paper reports (Sec. V).

    python examples/paper_harness.py                        # synthetic suite (what CI runs)
    python examples/paper_harness.py --markdown report.md --json report.json
    python examples/paper_harness.py --scannet scans/scene0011_00 scans/scene0050_00 \\
        --detector groundingdino --cache_detections         # Tables II-III on ScanNet (GPU once)
    python examples/paper_harness.py --scannet scans/scene0011_00 --detector offline   # reuse the cache
    python examples/paper_harness.py --capture data/my_capture          # Tables IV-V on a real capture
    python examples/paper_harness.py --capture data/their_capture --paper_capture   # the paper's own data

The paper's numbers live in ``config/paper_results.yaml``; the comparisons,
and why each is a reproduction, a floor or a claim, in
``semantic_mapping/paper_harness.py``. Every result the paper reports gets a
row in the report, "not run" (with the reason) when its data is missing.

**Synthetic suite** (always, unless ``--skip_synthetic``): the synthetic
change scene has the six objects of Table IV (bucket, cart and safety sign
introduced; plant, trash can and chair removed), so their detection and
change recall must reach the paper's real-world values, and the identity
behaviour of Sec. V-C (Fig. 3) must hold. Table V's claim that the full
system beats every ablation is checked on a stress version of the scene,
averaged over ``--seeds`` seeds: detector labels flicker between prompted
classes (what Eq. 10 fusion absorbs) and dark objects lose their depth in
some frames (what the 2D tracker bridges). The plain scene cannot separate
those modules. Sec. V-H's rates are floors measured on the same run.

**ScanNet** (``--scannet``): Tables II and III, pooled over the given scenes.
The paper does not list its scenes, so the comparison is only as close as
the scene choice; ``--tolerance_pp`` sets the percentage-point band that
counts as reproduced. ``--cache_detections`` stores each scene's detections
under ``<scene>/detections`` so reruns can use ``--detector offline``.

**Capture** (``--capture``): a sequence in the ``datasets.py`` layout with a
``scene_ground_truth.json`` whose Table IV objects carry the paper's labels.
Its values are floors and claims, or reproductions with ``--paper_capture``.

Exits 1 when a floor or claim fails (and, with ``--strict``, when a
reproduction misses its tolerance).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_example_dataset import generate_scene  # noqa: E402
from semantic_mapping import evaluation, paper_harness as harness  # noqa: E402
from semantic_mapping import segmentation_metrics as seg  # noqa: E402
from semantic_mapping.datasets import load_dataset, load_prompts, load_yaml_params  # noqa: E402
from semantic_mapping.detectors import build_detector  # noqa: E402
from semantic_mapping.detectors.offline import write_detections  # noqa: E402

PACKAGE_DIR = Path(__file__).resolve().parent.parent

STRESS_SCENE = {
    "label_flip_prob": 0.2,
    "depth_dropout_labels": ("trash can", "chair", "bucket", "cart"),
    "depth_dropout_prob": 0.5,
}
"""The Table V stress scene: one detection in five reports a confusable prompted
label, and four dark objects lose all depth in half the frames."""


def _detector(name: str, dataset, params: dict):
    if name == "offline":
        return build_detector("offline", detections_dir=dataset.detections_dir)
    return build_detector(name, **params.get(name, {}))


def _temporal_ground_truth(dataset):
    path = Path(dataset.data_dir) / "scene_ground_truth.json"
    return evaluation.load_ground_truth(path)[1] if path.exists() else None


def run_ablation(dataset, detector, prompts, params: dict, ground_truth, full_run=None) -> dict:
    """Table V: final-map precision / recall / F1 of each configuration on one sequence."""
    rows = {}
    for name, overrides in harness.TABLE5_OVERRIDES.items():
        run = full_run if (not overrides and full_run is not None) else harness.run_sequence(
            dataset, detector, prompts, {**params, **overrides}, ground_truth)
        rows[name] = harness.ablation_row(run)
    return rows


def synthetic_suite(reference: dict, params: dict, prompts, work_dir: Path, seeds: int) -> list[harness.Check]:
    scene = generate_scene(work_dir / "change_scene", verbose=False)
    dataset = load_dataset(scene)
    run = harness.run_sequence(dataset, _detector("offline", dataset, params), prompts, params,
                               _temporal_ground_truth(dataset))
    label = f"synthetic change scene ({len(dataset)} frames, {dataset.intrinsics.width}x{dataset.intrinsics.height})"
    checks = harness.table4_checks(run.evaluator, reference, label)
    checks += harness.identity_checks(run.evaluator, reference, label)
    checks += harness.runtime_checks(run.timings_ms, reference, label + ", CPU")

    runs = []
    for seed in range(seeds):
        stress = load_dataset(generate_scene(work_dir / f"stress_{seed}", seed=seed, verbose=False, **STRESS_SCENE))
        runs.append(run_ablation(stress, _detector("offline", stress, params), prompts, params,
                                 _temporal_ground_truth(stress)))
    stress_label = (f"synthetic stress scene, {seeds} seeds (label flicker {STRESS_SCENE['label_flip_prob']:.0%}, "
                    f"depth dropout {STRESS_SCENE['depth_dropout_prob']:.0%} on "
                    f"{len(STRESS_SCENE['depth_dropout_labels'])} objects)")
    checks += harness.table5_checks(runs, reference, stress_label)
    return checks


def capture_suite(reference: dict, params: dict, prompts, args) -> list[harness.Check]:
    dataset = load_dataset(args.capture, frame_skip=args.frame_skip)
    ground_truth = _temporal_ground_truth(dataset)
    if ground_truth is None:
        raise SystemExit(f"{args.capture} has no scene_ground_truth.json (Sec. V-D ground truth)")
    detector = _detector(args.detector, dataset, params)
    run = harness.run_sequence(dataset, detector, prompts, params, ground_truth, args.detector_rate_hz)
    label = f"capture {Path(args.capture).name} ({len(dataset)} frames)"
    checks = harness.table4_checks(run.evaluator, reference, label, reproduction=args.paper_capture,
                                   tolerance=args.tolerance)
    checks += harness.identity_checks(run.evaluator, reference, label)
    checks += harness.runtime_checks(run.timings_ms, reference, label)
    rows = run_ablation(dataset, detector, prompts, params, ground_truth, full_run=run)
    checks += harness.table5_checks([rows], reference, label, reproduction=args.paper_capture,
                                    tolerance=args.tolerance)
    return checks


def scannet_suite(reference: dict, params: dict, prompts, eval_cfg: dict, args) -> list[harness.Check]:
    aliases = eval_cfg.get("aliases") or None
    classes = [seg.normalize_label(c, aliases) for c in reference["table3"]["classes"]]
    scenes = []
    for scene_dir in args.scannet:
        dataset = load_dataset(scene_dir, frame_skip=args.frame_skip)
        gt = dataset.ground_truth_points()
        if gt is None:
            raise SystemExit(f"{scene_dir} has no annotated mesh (Sec. V-B ground truth)")
        detector = _detector(args.detector, dataset, params)
        caching = args.cache_detections and args.detector != "offline"

        def cache(frame, detections, directory=dataset.detections_dir):
            write_detections(directory, frame.frame_id, detections)

        run = harness.run_sequence(dataset, detector, prompts, params, detector_rate_hz=args.detector_rate_hz,
                                   on_frame=cache if caching else None)
        scenes.append((run.final_objects, gt))
        print(f"  {Path(scene_dir).name}: {len(dataset)} frames, {len(run.final_objects)} instances")
    report = seg.pooled_segmentation_report(
        scenes, background_classes=eval_cfg.get("background_classes", seg.DEFAULT_BACKGROUND_CLASSES),
        aliases=aliases, max_distance=float(eval_cfg.get("max_distance", seg.DEFAULT_MAX_DISTANCE_M)),
        instance_classes=classes)
    label = (f"ScanNet, {len(scenes)} scene{'s' if len(scenes) != 1 else ''} (frame_skip {args.frame_skip}, "
             f"detector {args.detector}; the paper used Grounding DINO + SAM2 on scenes it does not list)")
    return (harness.table2_checks(report, reference, label, args.tolerance_pp)
            + harness.table3_checks(report, reference, label, aliases, args.tolerance_pp))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=None, help="Paper results (default: config/paper_results.yaml).")
    parser.add_argument("--deviations", default=None,
                        help="Known deviations (default: config/paper_deviations.yaml); 'none' to report every miss.")
    parser.add_argument("--config", default=str(PACKAGE_DIR / "config" / "semantic_mapping.yaml"))
    parser.add_argument("--prompts", default=str(PACKAGE_DIR / "config" / "prompts.yaml"))
    parser.add_argument("--eval_config", default=str(PACKAGE_DIR / "config" / "segmentation_eval.yaml"))
    parser.add_argument("--skip_synthetic", action="store_true", help="Skip the synthetic suite.")
    parser.add_argument("--seeds", type=int, default=5, help="Stress-scene seeds for the Table V claim.")
    parser.add_argument("--work_dir", type=Path, default=None,
                        help="Where to write the synthetic scenes (default: a temporary directory).")
    parser.add_argument("--scannet", nargs="*", default=[], help="Annotated ScanNet scenes for Tables II-III.")
    parser.add_argument("--capture", default=None, help="A sequence with scene_ground_truth.json for Tables IV-V.")
    parser.add_argument("--paper_capture", action="store_true",
                        help="--capture is the paper's own data: compare values, not just floors and claims.")
    parser.add_argument("--detector", choices=["offline", "yoloe", "groundingdino"], default="offline",
                        help="Detector for --scannet / --capture (the paper: groundingdino with SAM2).")
    parser.add_argument("--cache_detections", action="store_true",
                        help="Store --scannet detections under <scene>/detections for --detector offline reruns "
                             "(with the same --frame_skip and --detector_rate_hz: only detected frames are stored).")
    parser.add_argument("--frame_skip", type=int, default=10, help="Use every N-th frame of --scannet / --capture.")
    parser.add_argument("--detector_rate_hz", type=float, default=0.0,
                        help="Detect at this rate on --scannet / --capture (paper: 1 Hz); 0 = every frame.")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="Band that counts as reproduced for recall, precision and F1.")
    parser.add_argument("--tolerance_pp", type=float, default=2.0,
                        help="Band in percentage points that counts as reproduced for Tables II-III.")
    parser.add_argument("--strict", action="store_true", help="Also fail when a reproduction misses its band.")
    parser.add_argument("--markdown", type=Path, default=None, help="Write the report here too.")
    parser.add_argument("--json", type=Path, default=None, help="Write the checks as JSON here too.")
    parser.add_argument("--append_summary", type=Path, default=None,
                        help="Append the report to this file (CI: $GITHUB_STEP_SUMMARY).")
    args = parser.parse_args()

    reference = harness.load_paper_results(args.results)
    params = load_yaml_params(args.config)
    prompts = load_prompts(args.prompts)
    eval_cfg = {}
    if Path(args.eval_config).exists():
        eval_cfg = yaml.safe_load(Path(args.eval_config).read_text()) or {}

    checks = harness.protocol_checks(reference)
    if args.scannet:
        print(f"ScanNet: {len(args.scannet)} scenes")
        checks += scannet_suite(reference, params, prompts, eval_cfg, args)
    else:
        checks += harness.table2_checks(None, reference, "") + harness.table3_checks(None, reference, "")
    if not args.skip_synthetic:
        print(f"Synthetic suite: change scene and {args.seeds} stress seeds")
        if args.work_dir is not None:
            checks += synthetic_suite(reference, params, prompts, args.work_dir, args.seeds)
        else:
            with tempfile.TemporaryDirectory(prefix="paper_harness_") as work_dir:
                checks += synthetic_suite(reference, params, prompts, Path(work_dir), args.seeds)
    if args.capture:
        print(f"Capture: {args.capture}")
        checks += capture_suite(reference, params, prompts, args)
    if args.skip_synthetic and not args.capture:
        for label in reference["table4"]["appeared"] + reference["table4"]["disappeared"]:
            for metric in ("detection", "change"):
                checks.append(harness.not_run(f"table4.{label}.{metric}", "Table IV", f"{label}: {metric} recall",
                                              reference["table4"]["methods"]["SuperMap"][label][metric],
                                              "needs --capture or the synthetic suite"))

    deviations = {} if args.deviations == "none" else harness.load_known_deviations(args.deviations)
    stale = harness.apply_known_deviations(checks, deviations, strict=args.strict)
    report = harness.format_report(checks, strict=args.strict)
    if stale:
        report += ("\n\nListed as known deviations but no longer missing the paper (remove from "
                   "config/paper_deviations.yaml): " + ", ".join(stale))
    title = "## SuperMap against its paper (Sec. V)\n\n"
    print("\n" + title + report)
    if args.markdown is not None:
        args.markdown.write_text(title + report + "\n")
    if args.json is not None:
        args.json.write_text(json.dumps(harness.to_json(checks), indent=2))
    if args.append_summary is not None:
        with open(args.append_summary, "a") as f:
            f.write(title + report + "\n")
    raise SystemExit(1 if any(check.failed(args.strict) for check in checks) else 0)


if __name__ == "__main__":
    main()
