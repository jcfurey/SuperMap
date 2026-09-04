from semantic_mapping import scene_graph as sg
from semantic_mapping import serialization
from semantic_mapping.types import ObjectStatus
from test.helpers import make_object


def test_serialize_frame_matches_documented_schema():
    table = make_object(1, "table", [0.0, 0.0, 0.0, 1.0, 1.0, 0.5])
    mug = make_object(2, "mug", [0.1, 0.1, 0.5, 0.9, 0.9, 0.7])
    mug.latest_stamp = 12.5
    graph = sg.build_scene_graph([table, mug])

    records = serialization.serialize_frame([table, mug], graph)

    assert len(records) == 2
    mug_record = next(r for r in records if r["id"] == 2)
    assert set(mug_record) == {"id", "label", "bbox3d", "center", "spatial_relations", "status", "latest_stamp",
                               "seconds_since_seen"}
    assert mug_record["label"] == "mug"
    assert mug_record["status"] == "active"
    assert mug_record["latest_stamp"] == 12.5
    assert {"predicate": "on", "target_id": 1} in mug_record["spatial_relations"]


def test_serialize_frame_json_round_trips():
    obj = make_object(1, "chair", [0.0, 0.0, 0.0, 1.0, 1.0, 0.8])
    graph = sg.build_scene_graph([obj])
    import json

    payload = json.loads(serialization.serialize_frame_json([obj], graph))
    assert payload[0]["id"] == 1


def test_serialize_frame_skips_nodes_missing_from_object_list():
    obj = make_object(1, "chair", [0.0, 0.0, 0.0, 1.0, 1.0, 0.8], status=ObjectStatus.DISAPPEARED)
    graph = sg.SceneGraph(node_ids=[1, 999], spatial_edges=[])
    records = serialization.serialize_frame([obj], graph)
    assert [r["id"] for r in records] == [1]


def test_seconds_since_seen_is_relative_to_now_or_the_freshest_instance():
    from semantic_mapping import scene_graph as sg
    from semantic_mapping.serialization import serialize_frame
    from semantic_mapping.types import ObjectStatus
    from test.helpers import make_object

    fresh = make_object(1, "table", [0, 0, 0, 1, 1, 1])
    fresh.latest_stamp = 50.0
    old = make_object(2, "plant", [3, 3, 0, 4, 4, 1], status=ObjectStatus.OCCLUDED)
    old.latest_stamp = 20.0
    graph = sg.build_scene_graph([fresh, old])
    by_id = {r["id"]: r for r in serialize_frame([fresh, old], graph)}
    assert by_id[1]["seconds_since_seen"] == 0.0 and by_id[2]["seconds_since_seen"] == 30.0
    by_id = {r["id"]: r for r in serialize_frame([fresh, old], graph, now=65.0)}
    assert by_id[1]["seconds_since_seen"] == 15.0 and by_id[2]["seconds_since_seen"] == 45.0
