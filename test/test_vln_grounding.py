import numpy as np

from semantic_mapping import scene_graph as sg
from semantic_mapping.vln.clients import KeywordVLMClient, ScriptedVLMClient
from semantic_mapping.vln.grounding import Grounder, select_local_subgraph
from test.helpers import make_object


def _scene():
    table = make_object(1, "table", [0.0, 0.0, 0.0, 1.0, 1.0, 0.5])
    mug = make_object(2, "mug", [0.1, 0.1, 0.5, 0.9, 0.9, 0.7])
    far_chair = make_object(3, "chair", [10.0, 10.0, 0.0, 10.5, 10.5, 0.9])
    objects = [table, mug, far_chair]
    return objects, sg.build_scene_graph(objects)


def test_ground_resolves_answer_ids_to_waypoints():
    objects, graph = _scene()
    grounder = Grounder(ScriptedVLMClient(["The mug is on the table. <answer>2</answer>"]))
    result = grounder.ground("go to the mug", objects, graph)
    assert result.ok
    assert result.target_ids == [2]
    assert np.allclose(result.waypoints[0], objects[1].center)
    assert "Instance 2 on Instance 1" in result.prompt
    assert result.to_dict()["waypoints"][0] == objects[1].center.tolist()


def test_ground_reports_missing_answer_and_unknown_ids():
    objects, graph = _scene()
    no_tags = Grounder(ScriptedVLMClient(["I don't know."])).ground("go", objects, graph)
    assert not no_tags.ok and "no <answer>" in no_tags.error

    unknown = Grounder(ScriptedVLMClient(["<answer>42</answer>"])).ground("go", objects, graph)
    assert not unknown.ok and unknown.unresolved_ids == [42]


def test_ground_surfaces_client_errors_instead_of_raising():
    objects, graph = _scene()
    result = Grounder(ScriptedVLMClient([])).ground("go", objects, graph)  # exhausted -> VLMError
    assert result.error and not result.ok


def test_local_subgraph_selection_drops_far_nodes_and_their_edges():
    objects, graph = _scene()
    nodes, sub = select_local_subgraph(objects, graph, center=np.zeros(3), radius=3.0)
    assert [n.instance_id for n in nodes] == [1, 2]
    assert all(e.subject_id in (1, 2) and e.object_id in (1, 2) for e in sub.spatial_edges)
    nodes, _ = select_local_subgraph(objects, graph, center=np.array([10.0, 10.0, 0.0]), max_objects=1)
    assert [n.instance_id for n in nodes] == [3]


def test_keyword_client_end_to_end_with_grounder():
    objects, graph = _scene()
    result = Grounder(KeywordVLMClient()).ground("go to the chair", objects, graph)
    assert result.ok and result.target_ids == [3]
