from itertools import permutations

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


def test_neighbor_search_skips_far_apart_pairs():
    near_a = make_object(1, "chair", [0.0, 0.0, 0.0, 0.4, 0.4, 0.8])
    far_b = make_object(2, "chair", [100.0, 100.0, 0.0, 100.4, 100.4, 0.8])
    edges = sg.build_spatial_edges([near_a, far_b], cluster_radius=2.0, beside_max_distance=50.0)
    assert edges == []  # outside the candidate pair radius


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


def test_on_emits_inverse_under_edge():
    table = make_object(1, "table", [0.0, 0.0, 0.0, 1.0, 1.0, 0.5])
    mug = make_object(2, "mug", [0.1, 0.1, 0.5, 0.9, 0.9, 0.7])
    edges = sg.build_spatial_edges([table, mug])
    assert sg.SpatialEdge(2, "on", 1) in edges
    assert sg.SpatialEdge(1, "under", 2) in edges


def test_on_is_class_dependent():
    # Geometrically a rug "supports" the table, but a rug is not a supporting class.
    rug = make_object(1, "rug", [0.0, 0.0, 0.0, 2.0, 2.0, 0.02])
    table = make_object(2, "table", [0.5, 0.5, 0.02, 1.5, 1.5, 0.8])
    assert sg.build_spatial_edges([rug, table]) == []
    # Purely geometric when no support classes are configured.
    assert sg.SpatialEdge(2, "on", 1) in sg.build_spatial_edges([rug, table], support_classes=())
    # And configurable: declare rugs as supports.
    assert sg.SpatialEdge(2, "on", 1) in sg.build_spatial_edges([rug, table], support_classes=("rug",))


def test_nearby_relations_survive_cluster_boundaries_and_input_order():
    objects = [make_object(i + 1, 'chair', [x - 0.05, 0, 0, x + 0.05, 0.1, 0.1])
               for i, x in enumerate([0.0, 1.9, 2.1])]
    expected = [(2, 'beside', 3), (3, 'beside', 2)]
    for scene in permutations(objects):
        assert _edge_set(sg.build_spatial_edges(list(scene))) == expected
    assert _edge_set(sg.build_spatial_edges(objects[1:])) == expected
    assert sg.build_spatial_edges([]) == []


def test_on_follows_predicate_parameters_and_small_boundary_crossings():
    table = make_object(1, 'table', [0, 0, 0, 1, 1, 0.5])
    mug = make_object(2, 'mug', [0.1, 0.1, 0.60001, 0.9, 0.9, 0.8])
    assert sg.build_spatial_edges([table, mug]) == []
    mug.bbox3d[2] = 0.59999
    assert sg.SpatialEdge(2, 'on', 1) in sg.build_spatial_edges([table, mug])
    assert sg.build_spatial_edges([table, mug], z_tolerance=0.09) == []
    assert sg.build_spatial_edges([table, mug], support_classes=['chair']) == []


def _edge_set(edges):
    return sorted((e.subject_id, e.predicate, e.object_id) for e in edges)


def test_edges_follow_the_scene_as_it_changes():
    table = make_object(1, "table", [0.0, 0.0, 0.0, 1.0, 1.0, 0.5])
    mug = make_object(2, "mug", [0.2, 0.2, 0.5, 0.6, 0.6, 0.6])
    chair = make_object(3, "chair", [1.2, 0.0, 0.0, 1.7, 0.5, 0.5])
    far = make_object(4, "lamp", [8.0, 8.0, 0.0, 8.2, 8.2, 1.0])

    scene = [table, mug, chair, far]
    first = _edge_set(sg.build_spatial_edges(scene))
    assert (2, "on", 1) in first and (1, "beside", 3) in first

    mug.bbox3d = np.array([3.0, 3.0, 0.0, 3.2, 3.2, 0.1])                 # mug moved off the table
    assert (2, "on", 1) not in _edge_set(sg.build_spatial_edges(scene))

    scene.append(make_object(5, "book", [0.3, 0.3, 0.5, 0.8, 0.8, 0.55]))  # a new object on the table
    assert (5, "on", 1) in _edge_set(sg.build_spatial_edges(scene))

    del scene[0]                                                          # the table is gone
    assert not any(p == "on" for _, p, _ in _edge_set(sg.build_spatial_edges(scene)))


def _scalar_edges(objects, cluster_radius=2.0, z_tolerance=0.1, xy_iou_threshold=0.05, beside_max_distance=1.0,
                  support_classes=sg.DEFAULT_SUPPORT_CLASSES, on_min_footprint_fraction=0.5):
    """The per-pair predicates build_spatial_edges evaluated before they were vectorised."""
    from semantic_mapping.geometry_utils import iou_xy

    def on(a, b):
        if abs(a.bbox3d[2] - b.bbox3d[5]) > z_tolerance:
            return False
        if iou_xy(a.bbox3d, b.bbox3d) > xy_iou_threshold:
            return True
        return on_min_footprint_fraction > 0 and sg.footprint_fraction(a.bbox3d, b.bbox3d) >= on_min_footprint_fraction

    def beside(a, b):
        return (abs(a.bbox3d[2] - b.bbox3d[2]) <= z_tolerance and iou_xy(a.bbox3d, b.bbox3d) <= xy_iou_threshold
                and float(np.linalg.norm(a.center[:2] - b.center[:2])) <= beside_max_distance)

    supports = set(support_classes) if support_classes else None
    objects = sorted(objects, key=lambda obj: obj.instance_id)
    edges = []
    for i, j in sg._neighbor_pairs(objects, cluster_radius):
        a, b = objects[i], objects[j]
        pair = []
        for subject, support in ((a, b), (b, a)):
            if (supports is None or support.label in supports) and on(subject, support):
                pair += [sg.SpatialEdge(subject.instance_id, "on", support.instance_id),
                         sg.SpatialEdge(support.instance_id, "under", subject.instance_id)]
        if not pair and beside(a, b):
            pair += [sg.SpatialEdge(a.instance_id, "beside", b.instance_id),
                     sg.SpatialEdge(b.instance_id, "beside", a.instance_id)]
        edges += pair
    return edges


def test_vectorised_relations_match_the_scalar_predicates():
    rng = np.random.default_rng(7)
    labels = ["table", "mug", "chair", "box", "lamp", "rug"]
    for trial in range(40):
        n = int(rng.integers(2, 60))
        # A coarse lattice makes exact ties common: tops level with bottoms,
        # centres exactly beside_max_distance apart, touching and zero-area boxes.
        low = rng.integers(0, 12, size=(n, 3)) * 0.25
        size = rng.integers(0, 5, size=(n, 3)) * 0.25
        boxes = np.hstack((low, low + size))
        if trial % 2:
            boxes += rng.normal(0.0, 1e-3, size=boxes.shape)
        objects = [make_object(i + 1, labels[int(rng.integers(len(labels)))], box) for i, box in enumerate(boxes)]
        for kwargs in ({}, {"on_min_footprint_fraction": 0.0}, {"support_classes": ()},
                       {"beside_max_distance": 0.5, "z_tolerance": 0.25}, {"xy_iou_threshold": 0.0},
                       {"xy_iou_threshold": 1 / 3, "on_min_footprint_fraction": 0.5, "support_classes": ()}):
            assert sg.build_spatial_edges(objects, **kwargs) == _scalar_edges(objects, **kwargs)


def test_beside_distance_ties_are_decided_like_the_scalar_norm(monkeypatch):
    # np.linalg.norm of a 2-vector is a BLAS dot product and can round one
    # unit in the last place away from sqrt(x*x + y*y), depending on the
    # CPU and BLAS build; a limit at the scalar distance must keep deciding
    # exactly as the scalar test did. The real norm is checked as it is,
    # then replaced by one that is always one unit off in either direction,
    # so the disagreement is exercised on every platform.
    def pair(corner):
        a = make_object(1, "chair", [0.0, 0.0, 0.0, 0.2, 0.2, 1.0])
        b = make_object(2, "chair", [corner[0], corner[1], 0.0, corner[0] + 0.2, corner[1] + 0.2, 1.0])
        difference = a.center[:2] - b.center[:2]
        return [a, b], float(np.sqrt(difference[0] * difference[0] + difference[1] * difference[1]))

    corners = np.random.default_rng(3).uniform(0.3, 0.8, size=(200, 2))
    for corner in corners:
        objects, _ = pair(corner)
        scalar = float(np.linalg.norm(objects[0].center[:2] - objects[1].center[:2]))
        for limit in (scalar, np.nextafter(scalar, 0.0)):
            assert sg.build_spatial_edges(objects, beside_max_distance=limit) == \
                _scalar_edges(objects, beside_max_distance=limit)

    real_norm = np.linalg.norm
    direction = [np.inf]

    def one_unit_off(x, *args, **kwargs):
        if args or kwargs or np.shape(x) != (2,):
            return real_norm(x, *args, **kwargs)
        return np.nextafter(np.sqrt(x[0] * x[0] + x[1] * x[1]), direction[0])

    monkeypatch.setattr(np.linalg, "norm", one_unit_off)
    decided_by_the_norm = 0
    for k, corner in enumerate(corners[:40]):
        direction[0] = np.inf if k % 2 else 0.0
        objects, elementwise = pair(corner)
        norm = float(np.linalg.norm(objects[0].center[:2] - objects[1].center[:2]))
        for limit in {elementwise, norm, np.nextafter(elementwise, 0.0), np.nextafter(norm, np.inf)}:
            expected = _scalar_edges(objects, beside_max_distance=limit)
            assert sg.build_spatial_edges(objects, beside_max_distance=limit) == expected
            decided_by_the_norm += bool(expected) != (elementwise <= limit)
    assert decided_by_the_norm == 40  # one limit per pair where the elementwise distance alone decides wrongly


def test_boxes_in_view_batches_the_single_box_test():
    from semantic_mapping.object_map import ObjectMap

    K = np.array([[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]])
    boxes = np.array([[-0.5, -0.5, 1.0, 0.5, 0.5, 2.0], [5.0, -0.5, 1.0, 6.0, 0.5, 2.0],
                      [-0.5, -0.5, -3.0, 0.5, 0.5, -2.0], [-0.5, -0.5, -1.0, 0.5, 0.5, 1.0]])
    batched = ObjectMap.boxes_in_view(boxes, K, np.eye(4), (120, 160))
    assert batched.tolist() == [True, False, False, True]
    assert batched.tolist() == [ObjectMap.may_be_in_view(b, K, np.eye(4), (120, 160)) for b in boxes]
