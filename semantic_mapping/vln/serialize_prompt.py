"""Spatio-temporal serialization and grounded waypoint extraction (Sec. IV-D).

The 4D scene graph is the interface between raw sensor data and a
Vision-Language Model: a local subgraph is serialized into structured text
(instance ID, label, 3D centroid, spatial/temporal relations) that a VLM can
reason over without ever seeing a point cloud or video frame. The VLM's
response is expected to name target instance IDs inside ``<answer>`` tags,
which are parsed back into 3D navigation waypoints via the map.
"""
from __future__ import annotations

import re

import numpy as np

from semantic_mapping.scene_graph import SceneGraph
from semantic_mapping.types import ObjectInstance, ObjectStatus

SCHEMA_PREAMBLE = """\
You are given a 4D scene graph of a robot's environment.
- Coordinate frame: right-handed, metric, world-fixed "{coordinate_frame}" frame (meters).
- Each node is one physical object instance: "Instance <id> (<label>) at [x, y, z]".
- Spatial edges describe the current layout, e.g. "Instance 3 on Instance 5".
- Temporal cues describe how an instance's state changed over time, e.g.
  "Instance 7 (plant) was last seen at [x, y, z] at t=12.4s and has since disappeared".
- Instance IDs are stable identities: the same ID always refers to the same physical object,
  even across occlusions, relocations, or long gaps in observation.

When answering, reason over the graph and put your final choice of target
instance ID(s) inside <answer></answer> tags, e.g. <answer>3</answer> or
<answer>3, 7</answer> for multiple targets. Do not put anything else inside
the tags.
"""

_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)

MOVED_THRESHOLD_M = 0.5
"""Displacement before an instance is described as having moved."""

SETTLE_SECONDS = 1.0
"""Trajectory samples younger than this (after first sight) are ignored when
judging motion: a freshly spawned object's centroid shifts as more of it is
observed and its 3D extent fills in, which is estimate refinement, not motion.
Reporting it as movement would hand the model a false temporal cue."""


def _format_center(center: np.ndarray) -> str:
    return f"[{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]"


def serialize_subgraph_to_text(
    objects: list[ObjectInstance],
    scene_graph: SceneGraph,
    include_temporal_cues: bool = True,
) -> str:
    """Render nodes, spatial edges, and temporal cues as structured text."""
    by_id = {obj.instance_id: obj for obj in objects}
    lines: list[str] = ["Nodes:"]

    for node_id in scene_graph.node_ids:
        obj = by_id.get(node_id)
        if obj is None:
            continue
        lines.append(f"  Instance {obj.instance_id} ({obj.label}) at {_format_center(obj.center)}")

    if scene_graph.spatial_edges:
        lines.append("Spatial relations:")
        for edge in scene_graph.spatial_edges:
            lines.append(f"  Instance {edge.subject_id} {edge.predicate} Instance {edge.object_id}")

    if include_temporal_cues:
        temporal_lines = []
        for node_id in scene_graph.node_ids:
            obj = by_id.get(node_id)
            if obj is None or not obj.trajectory:
                continue
            last_stamp, last_center, _last_status = obj.trajectory[-1]
            if obj.status == ObjectStatus.DISAPPEARED:
                temporal_lines.append(
                    f"  Instance {obj.instance_id} ({obj.label}) was last seen at "
                    f"{_format_center(last_center)} at t={last_stamp:.2f}s and has since disappeared."
                )
                continue
            statuses = [s[2] for s in obj.trajectory]
            if ObjectStatus.DISAPPEARED.value in statuses:
                gone = max(i for i, s in enumerate(statuses) if s == ObjectStatus.DISAPPEARED.value)
                gone_stamp = obj.trajectory[gone][0]
                before = next((obj.trajectory[i] for i in range(gone - 1, -1, -1)
                               if statuses[i] != ObjectStatus.DISAPPEARED.value), obj.trajectory[gone])
                back = next((s for s in obj.trajectory[gone + 1:] if s[2] != ObjectStatus.DISAPPEARED.value), None)
                if back is not None:
                    moved = float(np.linalg.norm(last_center - before[1])) > MOVED_THRESHOLD_M
                    where = (f"moved from {_format_center(before[1])} to {_format_center(last_center)}" if moved
                             else f"back in the same place at {_format_center(last_center)}")
                    temporal_lines.append(
                        f"  Instance {obj.instance_id} ({obj.label}) disappeared at t={gone_stamp:.2f}s and "
                        f"reappeared at t={back[0]:.2f}s, {where}."
                    )
                    continue
            settled = [s for s in obj.trajectory if s[0] >= obj.first_seen_stamp + SETTLE_SECONDS]
            if len(settled) < 2:
                continue
            ref_stamp, ref_center, _ = settled[0]
            if float(np.linalg.norm(last_center - ref_center)) > MOVED_THRESHOLD_M:
                temporal_lines.append(
                    f"  Instance {obj.instance_id} ({obj.label}) moved from "
                    f"{_format_center(ref_center)} at t={ref_stamp:.2f}s to "
                    f"{_format_center(last_center)} at t={last_stamp:.2f}s."
                )
        if temporal_lines:
            lines.append("Temporal cues:")
            lines.extend(temporal_lines)

    return "\n".join(lines)


def build_prompt(
    objects: list[ObjectInstance],
    scene_graph: SceneGraph,
    instruction: str,
    coordinate_frame: str = "map",
) -> str:
    """Assemble the full VLM prompt: schema + serialized graph + user instruction."""
    schema = SCHEMA_PREAMBLE.format(coordinate_frame=coordinate_frame)
    graph_text = serialize_subgraph_to_text(objects, scene_graph)
    return f"{schema}\n{graph_text}\n\nInstruction: {instruction}\n"


def parse_answer_ids(vlm_output: str) -> list[int]:
    """Extract target instance IDs from the VLM's ``<answer>...</answer>`` tags."""
    match = _ANSWER_PATTERN.search(vlm_output)
    if not match:
        return []
    ids: list[int] = []
    for token in re.split(r"[,\s]+", match.group(1).strip()):
        if token.isdigit():
            ids.append(int(token))
    return ids


def resolve_waypoints(instance_ids: list[int], objects: list[ObjectInstance]) -> list[np.ndarray]:
    """Map parsed instance IDs to 3D navigation waypoints (Sec. IV-D, grounded actuation)."""
    by_id = {obj.instance_id: obj for obj in objects}
    waypoints = []
    for instance_id in instance_ids:
        obj = by_id.get(instance_id)
        if obj is not None:
            waypoints.append(obj.center)
    return waypoints
