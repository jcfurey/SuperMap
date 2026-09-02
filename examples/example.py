#!/usr/bin/env python3
"""Run the SuperMap online pipeline offline against a recorded (or synthetic) sequence.

    python examples/prepare_example_dataset.py   # one-time: generate the demo sequence
    python examples/example.py                   # run the mapping pipeline
    python examples/evaluate.py                  # score it with the paper's metrics (Sec. V-D/V-E)

Options: --detector yoloe|offline|groundingdino, --data_dir <path>, --config <yaml>,
--prompts <yaml>, --live (rerun window).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_mapping.datasets import SequenceDataset, load_prompts, load_yaml_params, run_sequence  # noqa: E402
from semantic_mapping.detectors import build_detector  # noqa: E402
from semantic_mapping.pipeline import PipelineConfig, SemanticMappingPipeline  # noqa: E402
from semantic_mapping.serialization import serialize_frame  # noqa: E402


def _maybe_init_rerun(live: bool):
    if not live:
        return None
    try:
        import rerun as rr
    except ImportError:
        print("--live requested but the 'rerun-sdk' package is not installed; "
              "continuing headlessly. Install it with `pip install rerun-sdk` to visualize.")
        return None
    rr.init("supermap_example", spawn=True)
    return rr


def _log_to_rerun(rr, stamp: float, result) -> None:
    rr.set_time_seconds("stamp", stamp)
    for obj in result.objects:
        if obj.points_world.shape[0] == 0:
            continue
        rr.log(f"world/objects/{obj.instance_id}/points", rr.Points3D(obj.points_world))
        xmin, ymin, zmin, xmax, ymax, zmax = obj.bbox3d
        center = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
        half_size = [max((xmax - xmin) / 2, 1e-3), max((ymax - ymin) / 2, 1e-3), max((zmax - zmin) / 2, 1e-3)]
        rr.log(
            f"world/objects/{obj.instance_id}/box",
            rr.Boxes3D(centers=[center], half_sizes=[half_size],
                       labels=[f"{obj.instance_id}:{obj.label} ({obj.status.value})"]),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", choices=["offline", "yoloe", "groundingdino"], default="offline")
    parser.add_argument("--data_dir", default="data/example_scene")
    parser.add_argument("--config", default="config/semantic_mapping.yaml")
    parser.add_argument("--prompts", default="config/prompts.yaml")
    parser.add_argument("--live", action="store_true", help="Open a Rerun window to visualize the map live.")
    args = parser.parse_args()

    dataset = SequenceDataset(args.data_dir)
    params = load_yaml_params(args.config)
    prompts = load_prompts(args.prompts)

    if args.detector == "offline":
        detector = build_detector("offline", detections_dir=dataset.detections_dir)
    else:
        detector = build_detector(args.detector, **params.get(args.detector, {}))

    pipeline = SemanticMappingPipeline(PipelineConfig.from_dict(params))
    rr = _maybe_init_rerun(args.live)

    result = None
    for frame, detections, result in run_sequence(dataset, pipeline, detector, prompts):
        active = [o for o in result.objects if o.status.value == "active"]
        print(f"[t={frame.stamp:6.2f}s] frame {frame.frame_id:04d}: "
              f"{len(detections)} detections, {len(active)} active objects, "
              f"{len(result.scene_graph.spatial_edges)} spatial edges")
        if rr is not None:
            _log_to_rerun(rr, frame.stamp, result)

    print(f"\nProcessed {len(dataset)} frames.")
    if result is None:
        return

    # Both offline and live modes emit the same schema; this is what a downstream
    # consumer (VLM grounding, logging, evaluation) would receive per frame.
    last_frame_json = serialize_frame(result.objects, result.scene_graph)
    print(f"Example per-frame JSON record (last frame, {len(last_frame_json)} objects):")
    print(json.dumps(last_frame_json[:2], indent=2))

    print("\nFinal map contents:")
    for obj in sorted(pipeline.object_map.objects.values(), key=lambda o: o.instance_id):
        print(f"  #{obj.instance_id:>3} {obj.label:<14} status={obj.status.value:<11} "
              f"conf={obj.label_confidence:.2f} center={np.round(obj.center, 2).tolist()}")

    if (dataset.data_dir / "scene_ground_truth.json").exists():
        print("\nGround truth available: run examples/evaluate.py for the paper's metrics (Sec. V-D/V-E).")


if __name__ == "__main__":
    main()
