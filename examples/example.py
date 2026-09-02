#!/usr/bin/env python3
"""Run the SuperMap online pipeline offline against a recorded (or synthetic) sequence.

    python examples/prepare_example_dataset.py   # one-time: generate the demo sequence
    python examples/example.py                   # run the mapping pipeline

Options: --detector yoloe|offline, --data_dir <path>, --config <yaml>, --live (rerun window).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_mapping.detectors import build_detector  # noqa: E402
from semantic_mapping.pipeline import PipelineConfig, SemanticMappingPipeline  # noqa: E402
from semantic_mapping.serialization import serialize_frame  # noqa: E402
from semantic_mapping.types import CameraIntrinsics, Observation, StampedPose  # noqa: E402
from semantic_mapping.geometry_utils import se3_from_translation_quaternion  # noqa: E402


def _load_yaml_params(config_path: Path) -> dict:
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    for value in data.values():
        if isinstance(value, dict) and "ros__parameters" in value:
            return value["ros__parameters"]
    return data


def _load_prompts(prompts_path: Path) -> list[str]:
    with open(prompts_path) as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("prompts", []))


def _iter_frames(data_dir: Path):
    frames_dir = data_dir / "frames"
    frame_ids = sorted({p.name.split("_")[0] for p in frames_dir.glob("*_pose.json")})
    for frame_id_str in frame_ids:
        frame_id = int(frame_id_str)
        depth = np.load(frames_dir / f"{frame_id_str}_depth.npy")
        rgb = np.load(frames_dir / f"{frame_id_str}_rgb.npy")
        pose = json.loads((frames_dir / f"{frame_id_str}_pose.json").read_text())
        yield frame_id, rgb, depth, pose


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


def _summarize(pipeline: SemanticMappingPipeline, data_dir: Path) -> None:
    gt_path = data_dir / "scene_ground_truth.json"
    print("\nFinal map contents:")
    for obj in sorted(pipeline.object_map.objects.values(), key=lambda o: o.instance_id):
        print(f"  #{obj.instance_id:>3} {obj.label:<14} status={obj.status.value:<11} "
              f"conf={obj.label_confidence:.2f} center={np.round(obj.center, 2).tolist()}")

    if not gt_path.exists():
        return

    ground_truth = json.loads(gt_path.read_text())["objects"]
    known = list(pipeline.object_map.objects.values())
    print("\nScripted appearance/disappearance check (best-effort label match):")
    for gt in ground_truth:
        if gt["disappear_frac"] < 1.0:
            matches = [o for o in known if o.label == gt["label"]]
            ok = any(o.status.value == "disappeared" for o in matches)
            print(f"  {gt['label']:<14} scripted to disappear -> {'OK' if ok else 'NOT CONFIRMED'}")
        if gt["appear_frac"] > 0.0:
            matches = [o for o in known if o.label == gt["label"]]
            # Check the object was *ever* confirmed active, not just at the final
            # frame: a small, late-appearing object can leave the camera's view
            # again well before the sequence ends without that being a tracking
            # failure (mirrors the paper's recall-during-the-appearance-interval
            # metric, Sec. V-D, rather than a final-frame snapshot).
            ok = any(
                o.status.value == "active" or any(status == "active" for _, _, status in o.trajectory)
                for o in matches
            )
            print(f"  {gt['label']:<14} scripted to appear    -> {'OK' if ok else 'NOT CONFIRMED'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", choices=["offline", "yoloe", "groundingdino"], default="offline")
    parser.add_argument("--data_dir", default="data/example_scene")
    parser.add_argument("--config", default="config/semantic_mapping.yaml")
    parser.add_argument("--prompts", default="config/prompts.yaml")
    parser.add_argument("--live", action="store_true", help="Open a Rerun window to visualize the map live.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "intrinsics.json").exists():
        print(f"No dataset found at {data_dir}. Run examples/prepare_example_dataset.py first.")
        raise SystemExit(1)

    params = _load_yaml_params(Path(args.config))
    prompts = _load_prompts(Path(args.prompts))

    intrinsics_dict = json.loads((data_dir / "intrinsics.json").read_text())
    intrinsics = CameraIntrinsics(**intrinsics_dict)

    if args.detector == "offline":
        detector = build_detector("offline", detections_dir=data_dir / "detections")
    else:
        detector = build_detector(args.detector, **params.get(args.detector, {}))

    pipeline = SemanticMappingPipeline(PipelineConfig.from_dict(params))
    rr = _maybe_init_rerun(args.live)

    n_frames = 0
    for frame_id, rgb, depth, pose in _iter_frames(data_dir):
        T_world_from_cam = se3_from_translation_quaternion(
            np.array(pose["translation"]), np.array(pose["quaternion"]),
        )
        detections = detector.detect(rgb, prompts=prompts, frame_id=frame_id)

        observation = Observation(
            stamp=pose["stamp"],
            pose=StampedPose(stamp=pose["stamp"], T_world_from_frame=T_world_from_cam),
            intrinsics=intrinsics,
            rgb=rgb,
            depth=depth,
            detections=detections,
        )
        result = pipeline.process_frame(observation)
        n_frames += 1

        active = [o for o in result.objects if o.status.value == "active"]
        print(f"[t={pose['stamp']:6.2f}s] frame {frame_id:04d}: "
              f"{len(detections)} detections, {len(active)} active objects, "
              f"{len(result.scene_graph.spatial_edges)} spatial edges")

        if rr is not None:
            _log_to_rerun(rr, pose["stamp"], result)

    print(f"\nProcessed {n_frames} frames.")

    # Both offline and live modes emit the same schema; this is what a downstream
    # consumer (VLM grounding, logging, evaluation) would receive per frame.
    last_frame_json = serialize_frame(result.objects, result.scene_graph)
    print(f"Example per-frame JSON record (last frame, {len(last_frame_json)} objects):")
    print(json.dumps(last_frame_json[:2], indent=2))

    _summarize(pipeline, data_dir)


if __name__ == "__main__":
    main()
