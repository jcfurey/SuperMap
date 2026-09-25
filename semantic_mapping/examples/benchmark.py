#!/usr/bin/env python3
"""Measure per-module runtime and memory on a sequence (Sec. V-H).

    python examples/benchmark.py                          # synthetic scene, pre-baked detections
    python examples/benchmark.py --detector yoloe         # include a real detector's latency
    python examples/benchmark.py --data_dir scans/scene0000_00 --frame_skip 10 --json runtime.json

Reports, per stage of the map update (appearance embedding, tracklet prediction, back-projection,
association, map update, scene-graph construction) and for the detector,
the mean / median / p95 latency and the rate that latency sustains, plus
the map's footprint and the process's peak resident memory. The paper's
module rates (detector 1 Hz, 3D mapping 3 Hz, 4D scene graph 5 Hz) map onto
the ``detector``, ``total`` (everything but detection), and ``scene_graph``
rows. Note that latency scales with image resolution and map size, so
compare like with like: the synthetic scene is 160x120.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_mapping import runtime  # noqa: E402
from semantic_mapping.datasets import load_dataset, load_prompts, load_yaml_params  # noqa: E402
from semantic_mapping.detectors import build_detector  # noqa: E402
from semantic_mapping.pipeline import PipelineConfig, SemanticMappingPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", choices=["offline", "yoloe", "groundingdino"], default="offline")
    parser.add_argument("--data_dir", default="data/example_scene")
    parser.add_argument("--config", default="config/semantic_mapping.yaml")
    parser.add_argument("--prompts", default="config/prompts.yaml")
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=1, help="Run the sequence this many times, each with a fresh map.")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    dataset = load_dataset(args.data_dir, frame_skip=args.frame_skip, max_frames=args.max_frames)
    params = load_yaml_params(args.config)
    prompts = load_prompts(args.prompts)
    if args.detector == "offline":
        detector = build_detector("offline", detections_dir=dataset.detections_dir)
    else:
        detector = build_detector(args.detector, **params.get(args.detector, {}))

    stats = runtime.RuntimeStats()
    frames = 0
    wall_start = time.perf_counter()
    for _ in range(max(args.repeat, 1)):
        # Each replay starts a new timestamp epoch and an independent map.
        pipeline = SemanticMappingPipeline(PipelineConfig.from_dict(params))
        for frame in dataset:
            t0 = time.perf_counter()
            detections = detector.detect(frame.rgb, prompts=prompts, frame_id=frame.frame_id)
            stats.add("detector", time.perf_counter() - t0)
            result = pipeline.process_frame(dataset.observation(frame, detections))
            stats.add_timings(result.timings)
            frames += 1
    wall = time.perf_counter() - wall_start

    summary = stats.summary()
    memory = runtime.map_memory(pipeline.object_map.objects.values())
    memory["peak_rss_mb"] = runtime.peak_rss_mb()
    intr = dataset.intrinsics
    notes = (f"{frames} frames at {intr.width}x{intr.height} in {wall:.2f} s wall "
             f"({frames / wall:.1f} frames/s end to end, detector included)")
    print(runtime.format_runtime_summary(summary, memory, notes))

    if args.json is not None:
        args.json.write_text(json.dumps({
            "stages": summary, "memory": memory, "frames": frames, "wall_seconds": wall,
            "image_size": [intr.width, intr.height], "detector": args.detector, "data_dir": str(dataset.data_dir),
        }, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
