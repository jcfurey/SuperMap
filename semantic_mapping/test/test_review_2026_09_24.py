"""Regression tests for the core-algorithm findings of doc/review-2026-09-24.md.

Each test names the review ID it pins down.
"""
import copy
import json

import numpy as np
import pytest

from semantic_mapping import association, persistence
from semantic_mapping import evaluation as ev
from semantic_mapping import geometric_consistency as gc
from semantic_mapping import scene_graph as sg
from semantic_mapping import segmentation_metrics as sm
from semantic_mapping import semantic_fusion as sf
from semantic_mapping import tracking
from semantic_mapping.appearance import ColorHistogramEmbedder, cosine_similarity
from semantic_mapping.geometry_utils import centroid, iou_3d
from semantic_mapping.object_map import ObjectMap
from semantic_mapping.pipeline import PipelineConfig, SemanticMappingPipeline
from semantic_mapping.types import CameraIntrinsics, Detection2D, ObjectStatus, Observation, StampedPose
from test.helpers import make_object

INTRINSICS = CameraIntrinsics(fx=100.0, fy=100.0, cx=80.0, cy=60.0, width=160, height=120)
K = INTRINSICS.K
BOX = np.array([60.0, 40.0, 100.0, 80.0])


def _depth(distance: float | None = 2.0) -> np.ndarray:
    depth = np.full((120, 160), 8.0)
    if distance is not None:
        depth[40:80, 60:100] = distance
    return depth


def _obs(stamp, detections=(), depth=2.0, pose=None, rgb=None, evaluated=True):
    return Observation(
        stamp=stamp, pose=StampedPose(stamp=stamp, T_world_from_frame=np.eye(4) if pose is None else pose),
        intrinsics=INTRINSICS, depth=None if depth is False else _depth(depth), rgb=rgb,
        detections=list(detections), detections_evaluated=evaluated,
    )


def _det(label="chair", score=0.9, bbox=BOX):
    return Detection2D(bbox=np.array(bbox, dtype=np.float64), label=label, score=score)


def _looking_away() -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = np.diag([-1.0, 1.0, -1.0])  # rotated 180 degrees about y: the object is behind
    return pose


# ----------------------------------------------------------------------- C1
def test_c1_unmatched_frames_do_not_compound_the_prediction():
    pipeline = SemanticMappingPipeline()
    for i in range(3):
        pipeline.process_frame(_obs(i * 0.1, [_det()]))
    obj = next(iter(pipeline.object_map.objects.values()))
    after_match = copy.deepcopy(obj.track)
    for i in range(3, 33):  # 3 s without a detection, geometry still confirms the object
        pipeline.process_frame(_obs(i * 0.1))
    np.testing.assert_array_equal(obj.track.state, after_match.state)
    np.testing.assert_array_equal(obj.track.covariance, after_match.covariance)

    # The next frame's prior is one prediction over the whole 3 s gap.
    one_step = tracking.predict(after_match, 3.2 - obj.latest_stamp)
    compounded = after_match
    for _ in range(30):
        compounded = tracking.predict(compounded, 3.2 - obj.latest_stamp)
    assert one_step.covariance[0, 0] < compounded.covariance[0, 0] / 10
    result = pipeline.process_frame(_obs(3.3, [_det()]))
    assert result.detection_instance_ids == [obj.instance_id]


# ------------------------------------------------------------------ C2, C20
def test_c2_labels_gate_2d_association():
    track = tracking.init_track(BOX)
    beliefs = [{"chair": 1.0}]
    person = association.associate([track], [BOX], [BOX], track_label_beliefs=beliefs, detection_labels=["person"])
    chair = association.associate([track], [BOX], [BOX], track_label_beliefs=beliefs, detection_labels=["chair"])
    assert person.matches == [] and chair.matches == [(0, 0)]
    # Without labels the call stays purely geometric (backward compatible).
    assert association.associate([track], [BOX], [BOX]).matches == [(0, 0)]


def test_c2_label_cost_prefers_the_track_that_believes_the_label():
    tracks = [tracking.init_track(BOX), tracking.init_track(BOX)]
    beliefs = [{"chair": 0.3, "stool": 0.7}, {"chair": 0.9, "stool": 0.1}]
    result = association.associate(tracks, [BOX, BOX], [BOX], track_label_beliefs=beliefs,
                                   detection_labels=["chair"])
    assert result.matches == [(1, 0)]


def test_c2_person_in_front_of_chair_is_not_fused_into_it():
    pipeline = SemanticMappingPipeline()
    for i in range(3):
        pipeline.process_frame(_obs(i * 0.1, [_det()]))
    chair_id = next(iter(pipeline.object_map.objects))
    # The person stands 1 m in front of the chair. (A person detection lifting
    # onto the chair's own geometry is a relabelling for Eq. 10 to weigh;
    # see test_paper_review_2026_09_25.py, D1.)
    result = pipeline.process_frame(_obs(0.3, [_det("person")], depth=1.0))
    assert result.detection_instance_ids[0] != chair_id
    assert "person" not in pipeline.object_map.objects[chair_id].label_belief
    assert chair_id in pipeline.object_map.objects  # and no merge folded them together


def test_c20_negligible_label_mass_is_not_compatible():
    assert association.labels_compatible("chair", {"chair": 0.2}, 0.1)
    assert not association.labels_compatible("person", {"chair": 0.998, "person": 0.002}, 0.1)
    assert association.labels_compatible("person", {"chair": 0.998, "person": 0.002}, 0.0)  # old semantics

    m = ObjectMap()
    chair = m.spawn(BOX, np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]), "chair", 0.9, 0.0)
    chair.label_belief = {"chair": 0.995, "person": 0.005}  # one stray person detection
    person = m.spawn(BOX, np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]), "person", 0.9, 1.0)
    assert m.merge_duplicates() == []
    assert set(m.objects) == {chair.instance_id, person.instance_id}


# ----------------------------------------------------------------------- C3
def test_c3_one_contradicting_frame_does_not_prune_a_fresh_point():
    m = ObjectMap(tau_eps=0.1)
    obj = m.spawn(BOX, np.array([[0.0, 0.0, 2.0]]), "chair", 0.9, 0.0)
    obj.status = ObjectStatus.ACTIVE
    far = np.full((120, 160), 5.0)  # the point is seen through
    m.update_unmatched(obj, K, np.eye(4), far)
    assert obj.points_world.shape[0] == 1 and obj.point_log_odds[0] < 0
    m.update_unmatched(obj, K, np.eye(4), far)
    assert obj.points_world.shape[0] == 0
    assert m.effective_prune_log_odds < gc.logit(0.1)
    # prune_min_contradictions=1 restores the configured threshold.
    assert ObjectMap(prune_min_contradictions=1).effective_prune_log_odds == -1.5


def test_c3_silhouette_points_are_not_contradicted_by_the_background_pixel():
    depth = np.full((120, 160), 8.0)
    depth[:, :80] = 2.0  # object surface ends at column 79; column 80 is background
    point = np.array([[0.0045, 0.0, 2.0]])  # projects to u = 80.2 -> rounds onto the background
    states, _ = gc.classify_points(K, np.eye(4), depth, point, 0.15)
    assert states[0] == gc.GeometricState.DISAPPEARED  # the old nearest-pixel verdict
    states, _, _ = gc.project_and_classify(K, np.eye(4), depth, point, 0.15, contradiction_window_px=1)
    assert states[0] != gc.GeometricState.DISAPPEARED
    # A point with background all around is still seen through.
    states, _, _ = gc.project_and_classify(K, np.eye(4), depth, np.array([[0.5, 0.0, 2.0]]), 0.15,
                                           contradiction_window_px=1)
    assert states[0] == gc.GeometricState.DISAPPEARED


# ---------------------------------------------------------------------- C12
def test_c12_detector_misses_only_count_while_the_instance_is_observable():
    m = ObjectMap()
    obj = m.spawn(BOX, np.array([[0.0, 0.0, 2.0]]), "chair", 0.9, 0.0)
    m.update_unmatched(obj, K, np.eye(4), _depth(), in_view=False)
    assert obj.missed_detection_frames == 0
    m.update_unmatched(obj, K, np.eye(4), _depth(1.0))  # an occluder in front hides it
    assert obj.missed_detection_frames == 0
    m.update_unmatched(obj, K, np.eye(4), _depth(2.0))  # visible and undetected
    assert obj.missed_detection_frames == 1


def test_c12_object_seen_once_survives_a_camera_pan():
    pipeline = SemanticMappingPipeline(PipelineConfig(tentative_max_age=3))
    pipeline.process_frame(_obs(0.0, [_det()]))
    for i in range(1, 10):
        pipeline.process_frame(_obs(i * 0.1, pose=_looking_away()))
    obj = next(iter(pipeline.object_map.objects.values()))
    assert obj.status == ObjectStatus.TENTATIVE and obj.missed_detection_frames == 0


# ---------------------------------------------------------------------- C13
def test_c13_corroborated_frame_is_one_bayesian_update():
    m = ObjectMap()
    obj = m.spawn(BOX, np.array([[0.0, 0.0, 2.0]]), "chair", 0.9, 0.0)
    prior = {"chair": 0.6, "stool": 0.4}
    obj.label_belief = dict(prior)
    m.update_matched(obj, obj.track, np.zeros((0, 3)), _det(score=0.9), 0.1, K, np.eye(4), _depth())
    expected = sf.prune_low_confidence_labels(
        sf.bayesian_label_update(prior, "chair", 0.9, p_self=sf.P_SELF_CORROBORATED))
    assert obj.label_belief == pytest.approx(expected)
    squared = sf.bayesian_label_update(sf.bayesian_label_update(prior, "chair", 0.9), "chair", 0.9)
    assert obj.label_belief["stool"] > squared["stool"]


# ---------------------------------------------------------------------- C14
def test_c14_dark_pixels_carry_no_chroma():
    embedder = ColorHistogramEmbedder()
    rgb = np.zeros((40, 80, 3), dtype=np.uint8)
    rgb[:, 40:] = (0, 0, 255)
    black = _det(bbox=[0, 0, 40, 40])
    blue = _det(bbox=[40, 0, 80, 40])
    mostly_black = _det(bbox=[20, 0, 60, 40])  # half black, half blue
    e_black, e_blue, e_mixed = embedder.embed(rgb, [black, blue, mostly_black])
    assert e_black is None and e_blue is not None
    assert cosine_similarity(e_mixed, e_blue) == pytest.approx(1.0)  # the black half does not vote


# ---------------------------------------------------------------------- C15
def test_c15_neighbour_search_uses_object_extent():
    table = make_object(1, "table", [0.0, 0.0, 0.0, 6.0, 1.0, 0.8])
    monitor = make_object(2, "monitor", [5.1, 0.1, 0.8, 5.9, 0.9, 1.3])  # at the far end of a 6 m table
    far = make_object(3, "monitor", [9.0, 0.4, 0.8, 9.1, 0.5, 0.9])  # 3 m past the table's end
    assert np.linalg.norm(table.center - monitor.center) > 2.0
    assert sg._neighbor_pairs([table, monitor, far], 2.0) == [(0, 1)]
    edges = sg.build_spatial_edges([table, monitor, far], cluster_radius=2.0)
    assert any(e.subject_id == 2 and e.predicate == "on" and e.object_id == 1 for e in edges)
    assert not any(3 in (e.subject_id, e.object_id) for e in edges)


# ---------------------------------------------------------------------- C16
def test_c16_new_beliefs_are_normalized():
    m = ObjectMap()
    obj = m.spawn(BOX, np.zeros((0, 3)), "chair", 0.55, 0.0)
    assert sum(obj.label_belief.values()) == pytest.approx(1.0)
    assert obj.label_confidence == pytest.approx(sf.bayesian_label_update(obj.label_belief, "chair", 0.55)["chair"])


# ---------------------------------------------------------------------- C17
def test_c17_tentative_tracks_expire_without_depth():
    pipeline = SemanticMappingPipeline(PipelineConfig(tentative_max_age=3))
    pipeline.process_frame(_obs(0.0, [_det()], depth=False))
    obj = next(iter(pipeline.object_map.objects.values()))
    for i in range(1, 5):
        pipeline.process_frame(_obs(i * 0.1, depth=False))
    assert obj.status == ObjectStatus.DISAPPEARED
    # Geometry-only (not evaluated) frames still do not count.
    pipeline = SemanticMappingPipeline(PipelineConfig(tentative_max_age=3))
    pipeline.process_frame(_obs(0.0, [_det()], depth=False))
    for i in range(1, 8):
        pipeline.process_frame(_obs(i * 0.1, depth=False, evaluated=False))
    assert next(iter(pipeline.object_map.objects.values())).status == ObjectStatus.TENTATIVE


# ---------------------------------------------------------------------- C18
@pytest.mark.parametrize("params", [
    {"min_points_for_3d_association": 0}, {"max_points_per_object": 0}, {"min_hits_to_confirm": 0},
    {"prune_min_contradictions": 0}, {"contradiction_window_px": -1}, {"voxel_size": 0.0},
    {"label_compatibility_min_mass": 1.5}, {"reconcile_max_gap_sec": -1.0},
])
def test_c18_invalid_counts_are_rejected_up_front(params):
    with pytest.raises(ValueError):
        PipelineConfig(**params)


# ---------------------------------------------------------------------- C19
def test_c19_caller_detections_are_not_mutated():
    rgb = np.full((120, 160, 3), (200, 30, 30), dtype=np.uint8)
    detection = _det()
    result = SemanticMappingPipeline().process_frame(_obs(0.0, [detection], rgb=rgb))
    assert detection.embedding is None
    assert result.objects[0].embedding is not None


# ---------------------------------------------------------------------- C21
def _retired_and_candidate(distance: float, gap: float):
    m = ObjectMap()
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    old = m.spawn(BOX, np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]), "box", 0.9, 0.0, embedding=emb)
    old.latest_stamp = 10.0
    old.status = ObjectStatus.DISAPPEARED
    new = m.spawn(BOX, np.array([[distance, 0.0, 2.0], [distance + 0.1, 0.0, 2.0]]), "box", 0.9,
                  10.0 + gap, embedding=emb)
    return m, old, new


@pytest.mark.parametrize("distance, gap, reconciled", [
    (2.0, 5.0, True), (50.0, 5.0, False), (2.0, 3600.0, False),
])
def test_c21_reconciliation_needs_spatial_and_temporal_plausibility(distance, gap, reconciled):
    m, old, new = _retired_and_candidate(distance, gap)
    merged = m.reconcile_retired([old], min_similarity=0.85)
    assert (merged == [(old.instance_id, new.instance_id)]) is reconciled


# ---------------------------------------------------------------------- C22
def test_c22_point_cap_keeps_evidence_bearing_points():
    assert ObjectMap().max_points_per_object == PipelineConfig().max_points_per_object
    m = ObjectMap(max_points_per_object=10, voxel_size=0.05)
    existing = np.column_stack([np.arange(8) * 0.1, np.zeros(8), np.full(8, 2.0)])
    obj = m.spawn(BOX, existing, "sofa", 0.9, 0.0)
    obj.point_log_odds[:] = 2.0
    new = np.column_stack([np.arange(20) * 0.1 + 5.0, np.zeros(20), np.full(20, 2.0)])
    m._fuse_points(obj, new)
    assert obj.points_world.shape[0] == 10
    assert np.count_nonzero(obj.point_log_odds == 2.0) == 8  # all accumulated evidence survives


# ---------------------------------------------------------------------- C32
def test_c32_identity_consistency_uses_each_phases_majority_id():
    gts = [ev.GroundTruthObject("box", np.zeros(6), 0, 5, set(range(5)), identity="box#1"),
           ev.GroundTruthObject("box", np.zeros(6), 5, 10, set(range(5, 10)), identity="box#1")]
    evaluator = ev.SequenceEvaluator(gts, INTRINSICS)
    evaluator.stats[0].matched_id_frames = {1: 5}
    evaluator.stats[1].matched_id_frames = {1: 1, 2: 4}  # a fragment carried the phase
    assert evaluator.identity_consistency()["consistent"] == 0
    evaluator.stats[1].matched_id_frames = {1: 4, 2: 1}
    assert evaluator.identity_consistency()["consistent"] == 1


# ----------------------------------------------------------------- C11, C31
def test_c31_points_keep_full_precision_far_from_the_origin(tmp_path):
    m = ObjectMap()
    obj = make_object(1, "sign", [0, 0, 0, 1, 1, 1])
    obj.points_world = np.array([[512345.678, 5312345.013, 12.345]])
    obj.point_log_odds, obj.point_membership = np.zeros(1), np.zeros(1)
    m.objects = {1: obj}
    persistence.save_map(m, tmp_path / "map")
    restored = ObjectMap()
    persistence.load_map(tmp_path / "map", restored, resume=False)
    np.testing.assert_array_equal(restored.objects[1].points_world, obj.points_world)


def test_c31_legacy_float32_maps_still_load(tmp_path):
    m = ObjectMap()
    obj = make_object(1, "chair", [0, 0, 0, 1, 1, 1])
    obj.points_world, obj.point_log_odds, obj.point_membership = np.ones((2, 3)), np.zeros(2), np.zeros(2)
    m.objects = {1: obj}
    persistence.save_map(m, tmp_path / "map")
    header = json.loads((tmp_path / "map" / persistence.MAP_JSON).read_text())
    # Rewrite it in the pre-atomic layout: fixed arrays name, float32 points, no arrays_file.
    with np.load(tmp_path / "map" / header.pop("arrays_file")) as arrays:
        legacy = {k: (v.astype(np.float32)) for k, v in arrays.items()}
    for stale in (tmp_path / "map").glob("map_arrays*"):
        stale.unlink()
    np.savez_compressed(tmp_path / "map" / persistence.MAP_ARRAYS, **legacy)
    (tmp_path / "map" / persistence.MAP_JSON).write_text(json.dumps(header))
    restored = ObjectMap()
    persistence.load_map(tmp_path / "map", restored, resume=False)
    np.testing.assert_array_equal(restored.objects[1].points_world, np.ones((2, 3)))


def _one_object_map(x: float) -> ObjectMap:
    m = ObjectMap()
    obj = make_object(1, "chair", [x, 0, 0, x + 1, 1, 1])
    obj.points_world, obj.point_log_odds, obj.point_membership = np.full((1, 3), x), np.zeros(1), np.zeros(1)
    m.objects = {1: obj}
    return m


@pytest.mark.parametrize("fail_on", ["arrays", "json"])
def test_c11_interrupted_save_leaves_the_previous_map_loadable(tmp_path, monkeypatch, fail_on):
    persistence.save_map(_one_object_map(1.0), tmp_path / "map")
    real_replace = persistence.os.replace

    def crash(src, dst):
        if (fail_on == "json") == str(dst).endswith(persistence.MAP_JSON):
            raise OSError("simulated crash")
        return real_replace(src, dst)

    monkeypatch.setattr(persistence.os, "replace", crash)
    with pytest.raises(OSError):
        persistence.save_map(_one_object_map(2.0), tmp_path / "map")
    monkeypatch.undo()
    restored = ObjectMap()
    persistence.load_map(tmp_path / "map", restored, resume=False)
    assert restored.objects[1].points_world[0, 0] == 1.0
    assert not list((tmp_path / "map").glob(".*.tmp"))  # temporaries cleaned up

    # The next successful save commits and garbage-collects the orphaned arrays file.
    persistence.save_map(_one_object_map(3.0), tmp_path / "map")
    assert len(list((tmp_path / "map").glob("map_arrays*.npz"))) == 1
    persistence.load_map(tmp_path / "map", restored, resume=False)
    assert restored.objects[1].points_world[0, 0] == 3.0


# ---------------------------------------------------------------------- C37
@pytest.mark.parametrize("new_clock, rebased", [(5.0, True), (5000.0, False)])
def test_c37_reloaded_map_does_not_mix_clock_epochs(tmp_path, new_clock, rebased):
    first = SemanticMappingPipeline()
    for i in range(3):
        first.process_frame(_obs(1000.0 + i * 0.1, [_det()]))
    first.save(tmp_path / "map")
    saved = next(iter(first.object_map.objects.values()))

    second = SemanticMappingPipeline()
    second.load(tmp_path / "map")
    second.process_frame(_obs(new_clock))
    obj = second.object_map.objects[saved.instance_id]
    assert obj.latest_stamp < new_clock and obj.first_seen_stamp <= obj.latest_stamp
    assert all(stamp <= new_clock for stamp, _, _ in obj.trajectory)
    assert (obj.latest_stamp != saved.latest_stamp) is rebased
    assert obj.latest_stamp - obj.first_seen_stamp == pytest.approx(saved.latest_stamp - saved.first_seen_stamp)


# ----------------------------------------------------------------------- P8
def test_p8_matched_instances_reuse_the_batched_frustum_test(monkeypatch):
    pipeline = SemanticMappingPipeline()
    pipeline.process_frame(_obs(0.0, [_det()]))

    def per_instance_test(*_args, **_kwargs):
        raise AssertionError("per-instance frustum test repeated")

    monkeypatch.setattr(ObjectMap, "may_be_in_view", classmethod(per_instance_test))
    for i in range(1, 4):
        pipeline.process_frame(_obs(i * 0.1, [_det()] if i % 2 else []))


# ----------------------------------------------------------------------- P1
def _reidentify_reference(boxes, labels, embeddings, retired, min_similarity=0.85, iou_threshold=0.05,
                          margin=0.25, max_age_sec=0.0, now=0.0, candidates=None, min_mass=0.1):
    """The per-pair loop reidentify replaced, kept as the equivalence oracle."""
    def inside(point, box):
        return bool(np.all(point >= box[:3] - margin) and np.all(point <= box[3:] + margin))

    rows = list(range(len(retired)))
    cols = list(range(len(boxes))) if candidates is None else list(candidates)
    cost = np.full((len(rows), len(cols)), association.INVALID_COST)
    by_place = set()
    for r, i in enumerate(rows):
        obj = retired[i]
        if max_age_sec > 0 and now - obj.latest_stamp > max_age_sec:
            continue
        has_box = bool(np.any(obj.bbox3d[3:] > obj.bbox3d[:3]))
        for c, j in enumerate(cols):
            if boxes[j] is None or not association.labels_compatible(labels[j], obj.label_belief, min_mass):
                continue
            det_center = centroid(boxes[j])
            same_place = has_box and (iou_3d(boxes[j], obj.bbox3d) > iou_threshold
                                      or inside(det_center, obj.bbox3d) or inside(obj.center, boxes[j]))
            similarity = None
            if embeddings[j] is not None and obj.embedding is not None:
                similarity = cosine_similarity(embeddings[j], obj.embedding)
                if similarity < min_similarity:
                    continue
            if not same_place and similarity is None:
                continue
            appearance_cost = (1.0 - similarity) if similarity is not None else 0.5
            cost[r, c] = appearance_cost - (0.5 if same_place else 0.0) + 0.01 * float(
                np.linalg.norm(det_center - obj.center))
            if same_place:
                by_place.add((i, j))
    result = association._solve(cost, rows, cols, lambda i, j: True)
    return result, {pair for pair in by_place if pair in set(result.matches)}


@pytest.mark.parametrize("seed", range(12))
def test_p1_vectorized_reidentify_matches_the_pairwise_reference(seed):
    rng = np.random.default_rng(seed)
    labels = ["chair", "table", "box"]
    base = rng.normal(size=(4, 8))

    def embedding():
        if rng.random() < 0.2:
            return None
        if rng.random() < 0.05:
            return rng.normal(size=5).astype(np.float32)  # a descriptor of another dimension
        e = base[rng.integers(4)] + 0.2 * rng.normal(size=8)
        return (e / np.linalg.norm(e)).astype(np.float32)

    def box():
        lo = rng.uniform(-3, 3, size=3)
        return np.concatenate([lo, lo + rng.uniform(0.0 if rng.random() < 0.1 else 0.2, 1.0, size=3)])

    retired = []
    for i in range(int(rng.integers(0, 30))):
        obj = make_object(i + 1, labels[rng.integers(3)], box(), status=ObjectStatus.DISAPPEARED)
        if rng.random() < 0.3:
            obj.label_belief = {labels[0]: 0.95, labels[1]: 0.05}
        obj.embedding = embedding()
        obj.latest_stamp = float(rng.uniform(0, 100))
        retired.append(obj)
    n_det = int(rng.integers(0, 12))
    det_boxes = [None if rng.random() < 0.1 else box() for _ in range(n_det)]
    det_labels = [labels[rng.integers(3)] for _ in range(n_det)]
    det_embeddings = [embedding() for _ in range(n_det)]
    candidates = sorted(rng.choice(n_det, size=int(rng.integers(0, n_det + 1)), replace=False).tolist()) \
        if n_det and rng.random() < 0.5 else None
    kwargs = dict(min_similarity=0.8, max_age_sec=float(rng.choice([0.0, 50.0])), now=100.0)

    got, got_place = association.reidentify(det_boxes, det_labels, det_embeddings, retired,
                                            candidate_detections=candidates, **kwargs)
    want, want_place = _reidentify_reference(det_boxes, det_labels, det_embeddings, retired,
                                             candidates=candidates, **kwargs)
    assert got.matches == want.matches
    assert got.unmatched_tracks == want.unmatched_tracks
    assert got.unmatched_detections == want.unmatched_detections
    assert got_place == want_place


# ----------------------------------------------------------------------- P7
def _transfer_reference(objects, gt, max_distance, aliases=None):
    from scipy.spatial import cKDTree
    n_gt = gt.points.shape[0]
    pred_labels = np.full(n_gt, sm.UNLABELED, dtype=object)
    pred_ids = np.full(n_gt, -1, dtype=np.int64)
    with_points = [o for o in objects if o.status in ev.PRESENT_STATUSES and o.points_world.shape[0] > 0]
    if not with_points:
        return pred_labels.astype(str), pred_ids
    map_points = np.concatenate([o.points_world for o in with_points], axis=0)
    owners = np.concatenate([np.full(o.points_world.shape[0], o.instance_id) for o in with_points])
    distances, nearest = cKDTree(map_points).query(gt.points, distance_upper_bound=max_distance)
    found = np.isfinite(distances)
    pred_ids[found] = owners[nearest[found]]
    label_of = {o.instance_id: sm.normalize_label(o.label, aliases) for o in with_points}
    for instance_id in np.unique(pred_ids[found]):
        pred_labels[pred_ids == instance_id] = label_of[int(instance_id)]
    return pred_labels.astype(str), pred_ids


def _ap_reference(gt, transfer, thresholds, aliases=None):
    gt_labels = sm.normalize_labels(gt.labels, aliases)
    gt_instances = {}
    for gid in np.unique(gt.instance_ids[gt.instance_ids >= 0]):
        member = gt.instance_ids == gid
        labels, counts = np.unique(gt_labels[member], return_counts=True)
        gt_instances[int(gid)] = (str(labels[np.argmax(counts)]), member)
    preds = {pid: (sm.normalize_label(o.label, aliases), float(o.label_confidence), transfer.pred_instance_ids == pid)
             for pid, o in transfer.instances.items()}
    classes = sorted({name for name, _ in gt_instances.values() if name != sm.UNLABELED})
    out = {}
    for threshold in thresholds:
        per_class = {}
        for name in classes:
            gts = [g for g, (label, _) in gt_instances.items() if label == name]
            ranked = sorted((p for p, v in preds.items() if v[0] == name), key=lambda p: (-preds[p][1], p))
            matched, is_tp = set(), np.zeros(len(ranked), dtype=bool)
            for rank, pid in enumerate(ranked):
                best_iou, best = 0.0, None
                for g in gts:
                    if g in matched:
                        continue
                    inter = int(np.sum(preds[pid][2] & gt_instances[g][1]))
                    union = int(np.sum(preds[pid][2] | gt_instances[g][1]))
                    iou = inter / union if union else 0.0
                    if iou > best_iou:
                        best_iou, best = iou, g
                if best is not None and best_iou >= threshold:
                    matched.add(best)
                    is_tp[rank] = True
            per_class[name] = sm._average_precision(is_tp, len(gts))
        out[threshold] = per_class
    return classes, out


@pytest.mark.parametrize("seed", range(8))
def test_p7_vectorized_segmentation_metrics_match_the_reference(seed):
    rng = np.random.default_rng(seed)
    classes = ["chair", "table", "sofa", "wall"]
    n = 3000
    gt_ids = rng.integers(-1, 12, size=n)
    gt_label_of = {g: classes[rng.integers(4)] for g in range(12)}
    labels = np.array([gt_label_of[g] if g >= 0 else sm.UNLABELED for g in gt_ids], dtype=object)
    flip = rng.random(n) < 0.05  # a few mislabeled points make the majority vote matter
    labels[flip & (gt_ids >= 0)] = "chair"
    gt = sm.GroundTruthPoints(rng.uniform(0, 4, size=(n, 3)), labels.astype(str), gt_ids)
    objects = []
    for i in range(int(rng.integers(0, 15))):
        pts = gt.points[rng.choice(n, size=int(rng.integers(0, 300)), replace=False)] + rng.normal(0, 0.02, (1, 3))
        obj = make_object(100 + i, classes[rng.integers(4)], [0, 0, 0, 1, 1, 1],
                          status=[ObjectStatus.ACTIVE, ObjectStatus.DISAPPEARED][int(rng.random() < 0.2)])
        obj.label_belief = {obj.label: float(rng.choice([0.5, 0.7, 0.9]))}  # ties exercise the id tie-break
        obj.points_world = pts
        objects.append(obj)

    transfer = sm.transfer_labels(objects, gt, max_distance=0.05)
    want_labels, want_ids = _transfer_reference(objects, gt, 0.05)
    np.testing.assert_array_equal(transfer.pred_labels, want_labels)
    np.testing.assert_array_equal(transfer.pred_instance_ids, want_ids)

    report = sm.instance_level_ap(gt, transfer, (0.1, 0.25, 0.5))
    want_classes, want = _ap_reference(gt, transfer, (0.1, 0.25, 0.5))
    assert report["classes"] == want_classes
    for threshold, per_class in want.items():
        got = report[f"ap{int(round(threshold * 100))}"]["per_class"]
        assert got.keys() == per_class.keys()
        for name, value in per_class.items():
            assert (np.isnan(value) and np.isnan(got[name])) or got[name] == value
