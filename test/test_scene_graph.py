import numpy as np

from semantic_mapping import scene_graph as sg
from semantic_mapping.types import ObjectStatus
from test.helpers import make_object


def test_on_predicate_true_when_stacked_and_overlapping():
    table = make_object(1, "table", [0.0, 0.0, 0.0, 1.0, 1.0, 0.5])
    mug = make_object(2, "mug", [0.1, 0.1, 0.5, 0.9, 0.9, 0.7])
    edges = sg.build_spatial_edges([table, mug])
    assert sg.SpatialEdge(2, "on", 1) in edges


def test_on_predicate_false_when_not_overlapping_in_xy():
    table = make_object(1, "table", [0.0, 0.0, 0.0, 1.0, 1.0, 0.5])
    mug = make_object(2, "mug", [5.0, 5.0, 0.5, 5.2, 5.2, 0.7])
    edges = sg.build_spatial_edges([table, mug])
    assert edges == []


def test_beside_predicate_symmetric_for_similar_height_nearby_objects():
    chair_a = make_object(1, "chair", [0.0, 0.0, 0.0, 0.4, 0.4, 0.8])
    chair_b = make_object(2, "chair", [0.6, 0.0, 0.0, 1.0, 0.4, 0.8])
    edges = sg.build_spatial_edges([chair_a, chair_b], beside_max_distance=1.0)
    predicates = {(e.subject_id, e.predicate, e.object_id) for e in edges}
    assert (1, "beside", 2) in predicates
    assert (2, "beside", 1) in predicates


def test_clustering_skips_far_apart_pairs():
    near_a = make_object(1, "chair", [0.0, 0.0, 0.0, 0.4, 0.4, 0.8])
    far_b = make_object(2, "chair", [100.0, 100.0, 0.0, 100.4, 100.4, 0.8])
    edges = sg.build_spatial_edges([near_a, far_b], cluster_radius=2.0, beside_max_distance=50.0)
    assert edges == []  # never even compared: different clusters


def test_build_scene_graph_includes_disappeared_nodes_without_edges():
    active_obj = make_object(1, "table", [0.0, 0.0, 0.0, 1.0, 1.0, 0.5], status=ObjectStatus.ACTIVE)
    gone_obj = make_object(2, "plant", [0.3, 0.3, 0.5, 0.5, 0.5, 0.7], status=ObjectStatus.DISAPPEARED)
    graph = sg.build_scene_graph([active_obj, gone_obj])
    assert set(graph.node_ids) == {1, 2}
    assert all(edge.subject_id != 2 and edge.object_id != 2 for edge in graph.spatial_edges)


def test_record_trajectory_sample_skips_negligible_motion():
    obj = make_object(1, "table", [0.0, 0.0, 0.0, 1.0, 1.0, 0.5])
    sg.record_trajectory_sample(obj, stamp=0.0)
    sg.record_trajectory_sample(obj, stamp=0.1)  # same center, same status -> no new sample
    assert len(obj.trajectory) == 1

    obj.bbox3d = obj.bbox3d + np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # moved 1m in x
    sg.record_trajectory_sample(obj, stamp=0.2)
    assert len(obj.trajectory) == 2


def test_record_trajectory_sample_records_status_change_even_without_motion():
    obj = make_object(1, "table", [0.0, 0.0, 0.0, 1.0, 1.0, 0.5])
    sg.record_trajectory_sample(obj, stamp=0.0)
    obj.status = ObjectStatus.DISAPPEARED
    sg.record_trajectory_sample(obj, stamp=1.0)
    assert len(obj.trajectory) == 2
    assert obj.trajectory[-1][2] == "disappeared"
