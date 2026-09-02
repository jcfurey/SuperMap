#!/usr/bin/env python3
"""Score the pipeline on a sequence with ground truth using the paper's metrics.

    python examples/evaluate.py                       # defaults: data/example_scene, config/semantic_mapping.yaml
    python examples/evaluate.py --json results.json   # also dump machine-readable results

Implements Sec. V-D (object-detection recall and change-detection recall
with the paper's TP criterion: 3D IoU > 0.1, centroid < 0.3 m, correct
label) and a final-map precision/recall/F1 in the spirit of Sec. V-E, plus
an identity-fragmentation count. See semantic_mapping/evaluation.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_mapping import evaluation  # noqa: E402
from semantic_mapping.datasets import SequenceDataset, load_prompts, load_yaml_params, run_sequence  # noqa: E402
from semantic_mapping.detectors import build_detector  # noqa: E402
from semantic_mapping.pipeline import PipelineConfig, SemanticMappingPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", choices=["offline", "yoloe", "groundingdino"], default="offline")
    parser.add_argument("--data_dir", default="data/example_scene")
    parser.add_argument("--config", default="config/semantic_mapping.yaml")
    parser.add_argument("--prompts", default="config/prompts.yaml")
    parser.add_argument("--iou", type=float, default=evaluation.DEFAULT_IOU_THRESHOLD,
                        help="3D IoU threshold for a true positive (paper: 0.1).")
    parser.add_argument("--centroid", type=float, default=evaluation.DEFAULT_CENTROID_THRESHOLD_M,
                        help="Centroid distance threshold in meters for a true positive (paper: 0.3).")
    parser.add_argument("--json", type=Path, default=None, help="Write the summary to this JSON file too.")
    args = parser.parse_args()

    dataset = SequenceDataset(args.data_dir)
    gt_path = dataset.data_dir / "scene_ground_truth.json"
    if not gt_path.exists():
        print(f"No ground truth at {gt_path}; nothing to evaluate.")
        raise SystemExit(1)
    _num_frames, ground_truth = evaluation.load_ground_truth(gt_path)

    params = load_yaml_params(args.config)
    prompts = load_prompts(args.prompts)
    if args.detector == "offline":
        detector = build_detector("offline", detections_dir=dataset.detections_dir)
    else:
        detector = build_detector(args.detector, **params.get(args.detector, {}))
    pipeline = SemanticMappingPipeline(PipelineConfig.from_dict(params))

    evaluator = evaluation.SequenceEvaluator(
        ground_truth, dataset.intrinsics, iou_threshold=args.iou, centroid_threshold=args.centroid,
    )
    for frame, _detections, result in run_sequence(dataset, pipeline, detector, prompts):
        evaluator.observe(frame.frame_id, frame.T_world_from_cam, result.objects, depth_image=frame.depth)

    summary = evaluator.summary()
    print(f"Evaluated {len(dataset)} frames, {len(ground_truth)} ground-truth objects "
          f"(TP: 3D IoU > {args.iou}, centroid < {args.centroid} m, label match)\n")
    print(evaluation.format_summary(summary))

    if args.json is not None:
        args.json.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
