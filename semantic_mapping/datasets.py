"""Offline sequence loading shared by the example runner and the evaluator.

On-disk layout (what ``examples/prepare_example_dataset.py`` writes, and what a
real capture should be converted to)::

    <data_dir>/
      intrinsics.json               {"fx","fy","cx","cy","width","height"}
      frames/<frame_id:06d>_rgb.npy    (H, W, 3) uint8
      frames/<frame_id:06d>_depth.npy  (H, W) float32 meters, 0 = invalid
      frames/<frame_id:06d>_pose.json  {"stamp", "translation":[x,y,z], "quaternion":[x,y,z,w]}
                                        = world-from-camera pose P_t
      detections/<frame_id:06d>.json   pre-baked detections (see detectors/offline.py)
      scene_ground_truth.json          optional, see evaluation.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml

from semantic_mapping.detectors.base import Detector
from semantic_mapping.geometry_utils import se3_from_translation_quaternion
from semantic_mapping.pipeline import FrameResult, SemanticMappingPipeline
from semantic_mapping.types import CameraIntrinsics, Detection2D, Observation, StampedPose


@dataclass
class SequenceFrame:
    frame_id: int
    stamp: float
    rgb: np.ndarray
    depth: np.ndarray
    T_world_from_cam: np.ndarray


class SequenceDataset:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        intrinsics_path = self.data_dir / "intrinsics.json"
        if not intrinsics_path.exists():
            raise FileNotFoundError(
                f"No dataset at {self.data_dir} (missing intrinsics.json). "
                "Run examples/prepare_example_dataset.py first."
            )
        self.intrinsics = CameraIntrinsics(**json.loads(intrinsics_path.read_text()))
        self.frames_dir = self.data_dir / "frames"
        self.detections_dir = self.data_dir / "detections"
        self.frame_ids = sorted(int(p.name.split("_")[0]) for p in self.frames_dir.glob("*_pose.json"))

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __iter__(self) -> Iterator[SequenceFrame]:
        for frame_id in self.frame_ids:
            prefix = self.frames_dir / f"{frame_id:06d}"
            pose = json.loads(Path(f"{prefix}_pose.json").read_text())
            yield SequenceFrame(
                frame_id=frame_id,
                stamp=float(pose["stamp"]),
                rgb=np.load(f"{prefix}_rgb.npy"),
                depth=np.load(f"{prefix}_depth.npy"),
                T_world_from_cam=se3_from_translation_quaternion(
                    np.array(pose["translation"], dtype=np.float64),
                    np.array(pose["quaternion"], dtype=np.float64),
                ),
            )

    def observation(self, frame: SequenceFrame, detections: list[Detection2D]) -> Observation:
        return Observation(
            stamp=frame.stamp,
            pose=StampedPose(stamp=frame.stamp, T_world_from_frame=frame.T_world_from_cam),
            intrinsics=self.intrinsics,
            rgb=frame.rgb,
            depth=frame.depth,
            detections=detections,
        )


def run_sequence(
    dataset: SequenceDataset,
    pipeline: SemanticMappingPipeline,
    detector: Detector,
    prompts: list[str] | None,
) -> Iterator[tuple[SequenceFrame, list[Detection2D], FrameResult]]:
    """Drive the pipeline over a sequence, yielding each frame's inputs and result."""
    for frame in dataset:
        detections = detector.detect(frame.rgb, prompts=prompts, frame_id=frame.frame_id)
        result = pipeline.process_frame(dataset.observation(frame, detections))
        yield frame, detections, result


def load_yaml_params(config_path: str | Path) -> dict:
    """Load a ROS2-style params file, returning the ``ros__parameters`` block if present."""
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    for value in data.values():
        if isinstance(value, dict) and "ros__parameters" in value:
            return value["ros__parameters"]
    return data


def load_prompts(prompts_path: str | Path) -> list[str]:
    with open(prompts_path) as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("prompts", []))
