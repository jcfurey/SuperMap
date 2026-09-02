import numpy as np

from semantic_mapping import scene_graph as sg
from semantic_mapping.vln import serialize_prompt as vp
from test.helpers import make_object


def test_serialize_subgraph_lists_nodes_and_relations():
    table = make_object(1, "table", [0.0, 0.0, 0.0, 1.0, 1.0, 0.5])
    mug = make_object(2, "mug", [0.1, 0.1, 0.5, 0.9, 0.9, 0.7])
    graph = sg.build_scene_graph([table, mug])

    text = vp.serialize_subgraph_to_text([table, mug], graph)
    assert "Instance 1 (table)" in text
    assert "Instance 2 (mug)" in text
    assert "Instance 2 on Instance 1" in text


def test_build_prompt_includes_schema_and_instruction():
    obj = make_object(1, "chair", [0.0, 0.0, 0.0, 1.0, 1.0, 0.8])
    graph = sg.build_scene_graph([obj])
    prompt = vp.build_prompt([obj], graph, "Go to the chair.")
    assert "<answer>" in prompt
    assert "Go to the chair." in prompt
    assert "Instance 1 (chair)" in prompt


def test_parse_answer_ids_single_and_multiple():
    assert vp.parse_answer_ids("I think it's <answer>3</answer>") == [3]
    assert vp.parse_answer_ids("<answer>3, 7</answer>") == [3, 7]
    assert vp.parse_answer_ids("no answer tags here") == []


def test_resolve_waypoints_maps_ids_to_centers():
    obj = make_object(5, "chair", [0.0, 0.0, 0.0, 2.0, 2.0, 1.0])
    waypoints = vp.resolve_waypoints([5, 999], [obj])
    assert len(waypoints) == 1
    assert np.allclose(waypoints[0], [1.0, 1.0, 0.5])


def test_temporal_cue_mentions_disappearance():
    from semantic_mapping.types import ObjectStatus

    obj = make_object(1, "plant", [0.0, 0.0, 0.0, 0.2, 0.2, 0.5], status=ObjectStatus.DISAPPEARED)
    obj.trajectory = [(0.0, np.array([0.1, 0.1, 0.25]), "active"), (5.0, np.array([0.1, 0.1, 0.25]), "disappeared")]
    graph = sg.SceneGraph(node_ids=[1], spatial_edges=[])
    text = vp.serialize_subgraph_to_text([obj], graph)
    assert "has since disappeared" in text


def test_moved_cue_ignores_early_estimate_refinement_but_reports_real_motion():
    obj = make_object(1, "cart", [0.0, 0.0, 0.0, 0.5, 0.5, 1.0])
    obj.first_seen_stamp = 0.0
    # Centroid drifting 0.8 m within the first second: refinement, not motion.
    obj.trajectory = [(0.0, np.array([0.0, 0.0, 0.5]), "active"), (0.5, np.array([0.8, 0.0, 0.5]), "active"),
                      (1.5, np.array([0.85, 0.0, 0.5]), "active"), (3.0, np.array([0.9, 0.0, 0.5]), "active")]
    graph = sg.SceneGraph(node_ids=[1], spatial_edges=[])
    assert "moved" not in vp.serialize_subgraph_to_text([obj], graph)

    obj.trajectory.append((6.0, np.array([2.5, 0.0, 0.5]), "active"))  # genuinely relocated later
    text = vp.serialize_subgraph_to_text([obj], graph)
    assert "Instance 1 (cart) moved from [0.85, 0.00, 0.50] at t=1.50s to [2.50, 0.00, 0.50] at t=6.00s" in text
