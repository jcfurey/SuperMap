"""Runtime and memory accounting (Sec. V-H).

The paper reports per-module throughput: pose estimation at 10 Hz
(upstream), 2D instance segmentation at 1 Hz, 3D mapping at 3 Hz, and 4D
scene-graph updates at 5 Hz, all on the robot's onboard compute. This module
turns the per-frame stage timings that
:meth:`~semantic_mapping.pipeline.SemanticMappingPipeline.process_frame`
records (and a detector's measured latency) into the same kind of table:
mean / median / p95 latency and the sustainable rate per module, plus the
process's peak resident memory and the map's own footprint.
"""
from __future__ import annotations

import resource
import sys
from collections import defaultdict
from typing import Iterable

import numpy as np

from semantic_mapping.types import ObjectInstance

BYTES_PER_MAP_POINT = 3 * 8 + 8 + 8
"""float64 xyz + geometric log-odds + membership log-odds per stored point."""


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MiB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0  # bytes on macOS, KiB elsewhere


def map_memory(objects: Iterable[ObjectInstance]) -> dict:
    """Instance / point counts and the bytes the map's per-point arrays occupy."""
    objects = list(objects)
    points = sum(int(o.points_world.shape[0]) for o in objects)
    return {"instances": len(objects), "points": points, "bytes": points * BYTES_PER_MAP_POINT,
            "mb": points * BYTES_PER_MAP_POINT / (1024.0 * 1024.0)}


class RuntimeStats:
    """Accumulates latency samples per named stage."""

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = defaultdict(list)

    def add(self, stage: str, seconds: float) -> None:
        self._samples[stage].append(float(seconds))

    def add_timings(self, timings: dict[str, float]) -> None:
        for stage, seconds in timings.items():
            self.add(stage, seconds)

    def stages(self) -> list[str]:
        return list(self._samples)

    def summary(self) -> dict:
        """Per stage: sample count, mean / median / p95 / max latency in ms, and
        the rate (Hz) that mean latency sustains if the stage ran back to back."""
        out: dict[str, dict] = {}
        for stage, samples in self._samples.items():
            arr = np.asarray(samples, dtype=np.float64)
            if arr.size == 0:
                continue
            mean = float(arr.mean())
            out[stage] = {
                "count": int(arr.size),
                "mean_ms": 1e3 * mean,
                "p50_ms": 1e3 * float(np.percentile(arr, 50)),
                "p95_ms": 1e3 * float(np.percentile(arr, 95)),
                "max_ms": 1e3 * float(arr.max()),
                "hz": (1.0 / mean) if mean > 0 else float("inf"),
            }
        return out


def format_runtime_summary(stage_summary: dict, memory: dict | None = None, notes: str = "") -> str:
    lines = [f"{'stage':<16} {'n':>5} {'mean ms':>9} {'p50 ms':>8} {'p95 ms':>8} {'max ms':>8} {'rate Hz':>9}"]
    for stage, m in stage_summary.items():
        lines.append(f"{stage:<16} {m['count']:>5d} {m['mean_ms']:>9.2f} {m['p50_ms']:>8.2f} {m['p95_ms']:>8.2f} "
                     f"{m['max_ms']:>8.2f} {m['hz']:>9.1f}")
    if memory is not None:
        lines.append("")
        lines.append(f"map: {memory['instances']} instances, {memory['points']} points, {memory['mb']:.2f} MiB of point arrays")
        lines.append(f"process peak RSS: {memory['peak_rss_mb']:.0f} MiB")
    if notes:
        lines.append(notes)
    return "\n".join(lines)
