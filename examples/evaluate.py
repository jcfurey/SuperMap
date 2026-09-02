#!/usr/bin/env python3
"""Score the pipeline on a sequence with ground truth using the paper's metrics.

    python examples/evaluate.py                                        # synthetic scene, all metrics
    python examples/evaluate.py --data_dir scans/scene0000_00 --frame_skip 10 --detector yoloe   # ScanNet
    python examples/evaluate.py --json results.json                    # also dump machine-readable results

Which metrics run depends on the ground truth the sequence carries:

* ``scene_ground_truth.json`` -> Sec. V-D object-detection recall and
  change-detection recall (TP: 3D IoU > 0.1, centroid < 0.3 m, correct
  label), identity fragmentation, and a final-map precision / recall / F1 in
  the spirit of Sec. V-E (semantic_mapping/evaluation.py).
* ``gt_points.npz`` or an annotated ScanNet mesh -> Sec. V-B class-level
  mIoU / f-mIoU / accuracy with and without background (Table II) and
  instance-level AP25 / AP50 per class (Table III)
  (semantic_mapping/segmentation_metrics.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_mapping import evaluation, segmentation_metrics as seg  # noqa: E402
from semantic_mapping.datasets import load_dataset, load_prompts, load_yaml_params, run_sequence  # noqa: E402
from semantic_mapping.detectors import build_detector  # noqa: E402
from semantic_mapping.pipeline import PipelineConfig, SemanticMappingPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", choices=["offline", "yoloe", "groundingdino"], default="offline")
    parser.add_argument("--data_dir", default="data/example_scene",
                        help="Sequence directory (datasets.py layout) or a ScanNet scene export.")
    parser.add_argument("--config", default="config/semantic_mapping.yaml")
    parser.add_argument("--prompts", default="config/prompts.yaml")
    parser.add_argument("--eval_config", default="config/segmentation_eval.yaml",
                        help="Background classes, label aliases, and instance classes for the Sec. V-B metrics.")
    parser.add_argument("--frame_skip", type=int, default=1, help="Use every N-th frame (ScanNet is 30 Hz).")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--iou", type=float, default=evaluation.DEFAULT_IOU_THRESHOLD,
                        help="3D IoU threshold for a Sec. V-D true positive (paper: 0.1).")
    parser.add_argument("--centroid", type=float, default=evaluation.DEFAULT_CENTROID_THRESHOLD_M,
                        help="Centroid distance threshold in meters for a Sec. V-D true positive (paper: 0.3).")
    parser.add_argument("--max_distance", type=float, default=None,
                        help="Label-transfer radius in meters for the Sec. V-B metrics (overrides --eval_config).")
    parser.add_argument("--no_segmentation", action="store_true", help="Skip the Sec. V-B metrics.")
    parser.add_argument("--json", type=Path, default=None, help="Write all results to this JSON file too.")
    args = parser.parse_args()

    dataset = load_dataset(args.data_dir, frame_skip=args.frame_skip, max_frames=args.max_frames)
    temporal_gt_path = dataset.data_dir / "scene_ground_truth.json"
    temporal_gt = evaluation.load_ground_truth(temporal_gt_path)[1] if temporal_gt_path.exists() else None
    segmentation_gt = None if args.no_segmentation else dataset.ground_truth_points()
    if temporal_gt is None and segmentation_gt is None:
        print(f"No ground truth in {dataset.data_dir} (scene_ground_truth.json, gt_points.npz, or an "
              "annotated ScanNet mesh); nothing to evaluate.")
        raise SystemExit(1)

    params = load_yaml_params(args.config)
    prompts = load_prompts(args.prompts)
    if args.detector == "offline":
        detector = build_detector("offline", detections_dir=dataset.detections_dir)
    else:
        detector = build_detector(args.detector, **params.get(args.detector, {}))
    pipeline = SemanticMappingPipeline(PipelineConfig.from_dict(params))

    evaluator = None
    if temporal_gt is not None:
        evaluator = evaluation.SequenceEvaluator(
            temporal_gt, dataset.intrinsics, iou_threshold=args.iou, centroid_threshold=args.centroid,
        )
    result = None
    for frame, _detections, result in run_sequence(dataset, pipeline, detector, prompts):
        if evaluator is not None:
            evaluator.observe(frame.frame_id, frame.T_world_from_cam, result.objects, depth_image=frame.depth)

    results: dict = {"frames": len(dataset), "data_dir": str(dataset.data_dir)}
    print(f"Evaluated {len(dataset)} frames from {dataset.data_dir}")

    if evaluator is not None:
        results["temporal"] = evaluator.summary()
        print(f"\n== Spatio-temporal consistency (Sec. V-D / V-E): {len(temporal_gt)} ground-truth objects, "
              f"TP = 3D IoU > {args.iou}, centroid < {args.centroid} m, label match ==\n")
        print(evaluation.format_summary(results["temporal"]))

    if segmentation_gt is not None and result is not None:
        eval_cfg = {}
        if Path(args.eval_config).exists():
            with open(args.eval_config) as f:
                eval_cfg = yaml.safe_load(f) or {}
        report = seg.segmentation_report(
            result.objects, segmentation_gt,
            background_classes=eval_cfg.get("background_classes", seg.DEFAULT_BACKGROUND_CLASSES),
            aliases=eval_cfg.get("aliases") or None,
            max_distance=args.max_distance or float(eval_cfg.get("max_distance", seg.DEFAULT_MAX_DISTANCE_M)),
            instance_classes=eval_cfg.get("instance_classes") or None,
        )
        results["segmentation"] = report
        print(f"\n== Segmentation quality (Sec. V-B): {segmentation_gt.points.shape[0]} annotated points ==\n")
        print(seg.format_segmentation_report(report))

    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
