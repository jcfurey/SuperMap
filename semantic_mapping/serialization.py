"""Per-frame JSON serialization shared by offline and live (ROS2) modes.

Both entry points emit the same schema so downstream consumers (VLM
grounding, logging, evaluation) can use one shared interface regardless of
how the map was produced, as documented in the project README:
``bbox3d``, ``label``, ``id``, ``center``, ``spatial_relations``, ``status``,
and ``latest_stamp``.
"""
from __future__ import annotations

import json
from typing import Any

from semantic_mapping.scene_graph import SceneGraph
from semantic_mapping.types import ObjectInstance


def _relations_for(instance_id: int, scene_graph: SceneGraph) -> list[dict[str, Any]]:
    return [
        {"predicate": edge.predicate, "target_id": edge.object_id}
        for edge in scene_graph.spatial_edges
        if edge.subject_id == instance_id
    ]


def frame_time(objects: list[ObjectInstance], now: float | None = None) -> float:
    """The reference time for ages: ``now`` when given, else the freshest observation in the map."""
    if now is not None:
        return float(now)
    return max((obj.latest_stamp for obj in objects), default=0.0)


def serialize_instance(instance: ObjectInstance, scene_graph: SceneGraph, now: float | None = None) -> dict[str, Any]:
    """Serialize a single object instance to the documented per-frame schema.

    ``seconds_since_seen`` is how long ago the instance was last observed
    (relative to ``now``); consumers can treat a long-unobserved "occluded"
    instance as uncertain rather than present.
    """
    reference = frame_time([instance], now)
    return {
        "id": instance.instance_id,
        "label": instance.label,
        "bbox3d": instance.bbox3d.tolist(),
        "center": instance.center.tolist(),
        "spatial_relations": _relations_for(instance.instance_id, scene_graph),
        "status": instance.status.value,
        "latest_stamp": instance.latest_stamp,
        "seconds_since_seen": max(reference - instance.latest_stamp, 0.0),
    }


def serialize_frame(
    objects: list[ObjectInstance], scene_graph: SceneGraph, now: float | None = None,
) -> list[dict[str, Any]]:
    """Serialize every node currently in the scene graph, in a stable id order."""
    by_id = {obj.instance_id: obj for obj in objects}
    reference = frame_time(objects, now)
    return [
        serialize_instance(by_id[node_id], scene_graph, reference)
        for node_id in scene_graph.node_ids
        if node_id in by_id
    ]


def serialize_frame_json(
    objects: list[ObjectInstance], scene_graph: SceneGraph, now: float | None = None, **json_kwargs: Any,
) -> str:
    return json.dumps(serialize_frame(objects, scene_graph, now), **json_kwargs)
