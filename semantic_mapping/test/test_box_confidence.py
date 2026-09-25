"""Two-sided box support (trim and expand) and per-instance existence culling."""
import numpy as np
import pytest

from semantic_mapping.object_map import ObjectMap
from semantic_mapping.persistence import load_map, save_map
from semantic_mapping.pipeline import PipelineConfig, SemanticMappingPipeline
from semantic_mapping.types import Detection2D, ObjectStatus

W, H = 160, 120
K = np.array([[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]])
# Camera frame == world frame: +z is forward, so x/y offsets at z=10 m move
# 10 px per metre across the 160x120 image.
T = np.eye(4)
BBOX = np.array([60.0, 40.0, 100.0, 80.0])


def _cloud(center, n=60, spread=0.3, seed=0):
    return np.asarray(center, dtype=float) + np.random.default_rng(seed).uniform(-spread, spread, size=(n, 3))


def _map(**kw):
    kw.setdefault('bbox_min_support', 2)
    kw.setdefault('bbox_support_radius_m', 0.3)
    return ObjectMap(**kw)


def _match(om, obj, points, stamp, depth=None, score=0.9):
    detection = Detection2D(bbox=BBOX, label='bulk bag', score=score)
    depth = np.zeros((H, W)) if depth is None else depth
    om.update_matched(obj, obj.track, points, detection, stamp, K, T, depth)


def _spawn(om, points, score=0.9):
    return om.spawn(BBOX, points, 'bulk bag', score, 0.0)


BODY = _cloud([0.0, 0.0, 10.0])
STRAY = _cloud([2.0, 0.0, 10.0], n=20, spread=0.2, seed=1)  # in view, 20 px right of the body


def _rematch_body(om, obj, frames=4, depth=None):
    for i in range(1, frames + 1):
        _match(om, obj, BODY + np.random.default_rng(10 + i).normal(0, 0.02, BODY.shape), 0.1 * i, depth=depth)


def test_a_stray_region_views_no_longer_confirm_is_trimmed_and_culled():
    om = _map(bbox_support_miss=0.5)
    obj = _spawn(om, np.vstack((BODY, STRAY)))
    assert obj.bbox3d[3] > 2.0
    _rematch_body(om, obj)
    assert obj.bbox3d[3] < 0.6  # box trimmed back to the body
    assert np.all(obj.points_world[:, 0] < 1.0)  # stray points removed, not just hidden
    assert om.stats['support_culled_points'] >= 1


def test_support_only_grows_without_a_miss_penalty():
    om = _map()
    obj = _spawn(om, np.vstack((BODY, STRAY)))
    _rematch_body(om, obj)
    assert np.any(obj.points_world[:, 0] > 1.5)  # stray points kept (box still trims via min_support)
    assert om.stats['support_culled_points'] == 0


def test_occluded_and_out_of_image_points_keep_their_support():
    depth = np.zeros((H, W))
    depth[45:75, 95:115] = 5.0  # a nearer surface in front of the stray region
    om = _map(bbox_support_miss=0.5)
    obj = _spawn(om, np.vstack((BODY, STRAY)))
    _rematch_body(om, obj, depth=depth)
    assert np.any(obj.points_world[:, 0] > 1.5)  # occluded, so never charged a miss

    outside = _cloud([12.0, 0.0, 10.0], n=20, spread=0.2, seed=2)  # projects past the right edge
    om = _map(bbox_support_miss=0.5)
    obj = _spawn(om, np.vstack((BODY, outside)))
    _rematch_body(om, obj)
    assert np.any(obj.points_world[:, 0] > 11.0)


def test_a_region_repeatedly_observed_later_expands_the_box():
    extension = _cloud([0.0, 1.0, 10.0], n=30, spread=0.3, seed=3)
    om = _map(bbox_min_support=3, bbox_support_miss=0.5)
    obj = _spawn(om, BODY)
    _rematch_body(om, obj, frames=3)
    before = obj.bbox3d.copy()
    for i in range(3):
        _match(om, obj, np.vstack((BODY, extension)), 1.0 + 0.1 * i)
    assert obj.bbox3d[4] > before[4] + 0.8  # grew once the new region reached min support


def test_support_cap_lets_a_long_supported_region_decay():
    om = _map(bbox_support_miss=1.0, bbox_support_max=3.0)
    obj = _spawn(om, np.vstack((BODY, STRAY)))
    for i in range(6):  # stray region well supported for a while
        _match(om, obj, np.vstack((BODY, STRAY)), 0.1 * (i + 1))
    assert obj.point_support.max() == 3.0
    _rematch_body(om, obj, frames=4)
    assert np.all(obj.points_world[:, 0] < 1.0)


def _miss(om, obj, n, in_view=True):
    for _ in range(n):
        om.update_unmatched(obj, K, T, None, in_view=in_view, detections_evaluated=True)


def test_existence_culls_a_confirmed_box_the_detector_keeps_missing():
    om = ObjectMap(existence_hit_gain=1.0, existence_miss_penalty=1.0, existence_cull_log_odds=-2.0)
    obj = _spawn(om, BODY, score=0.9)
    obj.status = ObjectStatus.ACTIVE
    assert obj.existence_log_odds == pytest.approx(0.9)
    _miss(om, obj, 2)
    assert obj.status == ObjectStatus.ACTIVE
    _miss(om, obj, 1)
    assert obj.status == ObjectStatus.DISAPPEARED and om.stats['existence_culled'] == 1


def test_existence_rises_with_confident_hits_is_capped_and_ignores_unseen_frames():
    om = ObjectMap(existence_hit_gain=1.0, existence_max_log_odds=4.0)
    obj = _spawn(om, BODY, score=0.9)
    for i in range(10):
        _match(om, obj, BODY, 0.1 * (i + 1), score=0.8)
    assert obj.existence_log_odds == pytest.approx(4.0)
    obj.status = ObjectStatus.ACTIVE
    _miss(om, obj, 20, in_view=False)  # out of view: no evidence either way
    assert obj.existence_log_odds == pytest.approx(4.0) and obj.status == ObjectStatus.ACTIVE
    weak = ObjectMap(existence_hit_gain=1.0)
    low = _spawn(weak, BODY, score=0.3)
    assert low.existence_log_odds == pytest.approx(0.3)  # weak detections earn less confidence


def test_existence_is_off_by_default_and_round_trips(tmp_path):
    om = ObjectMap()
    obj = _spawn(om, BODY)
    obj.status = ObjectStatus.ACTIVE
    _miss(om, obj, 50)
    assert obj.existence_log_odds == 0.0 and obj.status == ObjectStatus.ACTIVE

    om = ObjectMap(existence_hit_gain=2.0)
    obj = _spawn(om, BODY, score=0.75)
    save_map(om, tmp_path)
    restored = ObjectMap(existence_hit_gain=2.0)
    load_map(tmp_path, restored, resume=False)
    assert restored.objects[obj.instance_id].existence_log_odds == pytest.approx(1.5)


def test_merges_keep_the_higher_existence():
    om = ObjectMap(existence_hit_gain=1.0)
    a = _spawn(om, BODY, score=0.4)
    b = _spawn(om, BODY + [0.01, 0, 0], score=0.9)
    assert om.merge_duplicates() == [(a.instance_id, b.instance_id)]
    assert a.existence_log_odds == pytest.approx(0.9)


@pytest.mark.parametrize('bad', [
    dict(bbox_support_miss=-1.0), dict(existence_hit_gain=float('nan')),
    dict(existence_cull_log_odds=6.0, existence_max_log_odds=6.0),
    dict(bbox_min_support=3, bbox_support_max=2.0),
])
def test_invalid_confidence_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        PipelineConfig(**bad)


def test_pipeline_passes_the_settings_through():
    om = SemanticMappingPipeline(PipelineConfig(
        bbox_min_support=2, bbox_support_miss=0.5, bbox_support_max=5.0, existence_hit_gain=1.5)).object_map
    assert (om.bbox_support_miss, om.bbox_support_max, om.existence_hit_gain) == (0.5, 5.0, 1.5)
