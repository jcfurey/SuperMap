"""Language-to-waypoint grounding over the 4D scene graph (Sec. IV-D).

Two-phase so the ROS node can snapshot the map on its executor thread and do
the (slow, network-bound) model call elsewhere:

1. :meth:`Grounder.prepare` -- select a local subgraph, serialize it, and
   capture each candidate's centroid.
2. :meth:`Grounder.complete` -- query the model, parse the ``<answer>`` tags,
   and resolve the chosen instance IDs to 3D waypoints from the snapshot.

:meth:`Grounder.ground` runs both for offline use.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from semantic_mapping.scene_graph import SceneGraph
from semantic_mapping.types import ObjectInstance
from semantic_mapping.vln.clients import VLMClient, VLMError
from semantic_mapping.vln.serialize_prompt import build_prompt, parse_answer_ids


@dataclass
class GroundingRequest:
    instruction: str
    prompt: str
    centers_by_id: dict[int, np.ndarray]
    labels_by_id: dict[str, str] = field(default_factory=dict)


@dataclass
class GroundingResult:
    instruction: str
    prompt: str
    response: str
    target_ids: list[int]
    waypoints: list[np.ndarray]
    unresolved_ids: list[int] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.waypoints)

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "target_ids": self.target_ids,
            "waypoints": [w.tolist() for w in self.waypoints],
            "unresolved_ids": self.unresolved_ids,
            "response": self.response,
            "error": self.error,
        }


def select_local_subgraph(
    objects: list[ObjectInstance],
    scene_graph: SceneGraph,
    center: np.ndarray | None = None,
    radius: float | None = None,
    max_objects: int | None = None,
) -> tuple[list[ObjectInstance], SceneGraph]:
    """Restrict the graph to nodes near ``center`` (the paper serializes a *local*
    subgraph): within ``radius`` meters and/or the ``max_objects`` nearest.
    With neither given, the whole graph is returned unchanged.
    """
    by_id = {obj.instance_id: obj for obj in objects}
    nodes = [by_id[i] for i in scene_graph.node_ids if i in by_id]
    if center is not None and (radius is not None or max_objects is not None):
        nodes.sort(key=lambda o: float(np.linalg.norm(o.center - center)))
        if radius is not None:
            nodes = [o for o in nodes if float(np.linalg.norm(o.center - center)) <= radius]
    if max_objects is not None:
        nodes = nodes[:max_objects]
    keep = {o.instance_id for o in nodes}
    edges = [e for e in scene_graph.spatial_edges if e.subject_id in keep and e.object_id in keep]
    return nodes, SceneGraph(node_ids=[o.instance_id for o in nodes], spatial_edges=edges)


class Grounder:
    def __init__(
        self,
        client: VLMClient,
        coordinate_frame: str = "map",
        local_radius_m: float | None = None,
        max_objects: int | None = None,
    ) -> None:
        self.client = client
        self.coordinate_frame = coordinate_frame
        self.local_radius_m = local_radius_m
        self.max_objects = max_objects

    def prepare(
        self,
        instruction: str,
        objects: list[ObjectInstance],
        scene_graph: SceneGraph,
        robot_position: np.ndarray | None = None,
    ) -> GroundingRequest:
        nodes, subgraph = select_local_subgraph(
            objects, scene_graph, robot_position, self.local_radius_m, self.max_objects,
        )
        prompt = build_prompt(nodes, subgraph, instruction, self.coordinate_frame)
        return GroundingRequest(
            instruction=instruction,
            prompt=prompt,
            centers_by_id={o.instance_id: o.center.copy() for o in nodes},
            labels_by_id={str(o.instance_id): o.label for o in nodes},
        )

    def complete(self, request: GroundingRequest) -> GroundingResult:
        try:
            response = self.client.complete(request.prompt)
        except VLMError as exc:
            return GroundingResult(request.instruction, request.prompt, "", [], [], error=str(exc))
        target_ids = parse_answer_ids(response)
        waypoints = [request.centers_by_id[i] for i in target_ids if i in request.centers_by_id]
        unresolved = [i for i in target_ids if i not in request.centers_by_id]
        error = None
        if not target_ids:
            error = "response contained no <answer> instance IDs"
        elif not waypoints:
            error = f"answered instance IDs {target_ids} are not in the map"
        return GroundingResult(request.instruction, request.prompt, response, target_ids, waypoints, unresolved, error)

    def ground(
        self,
        instruction: str,
        objects: list[ObjectInstance],
        scene_graph: SceneGraph,
        robot_position: np.ndarray | None = None,
    ) -> GroundingResult:
        return self.complete(self.prepare(instruction, objects, scene_graph, robot_position))
