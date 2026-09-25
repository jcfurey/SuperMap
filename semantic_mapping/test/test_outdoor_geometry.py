"""Outdoor sparse-LiDAR geometry options: ground removal, layer choice,
support-based bounds, and per-class size limits."""
import numpy as np
import pytest

from semantic_mapping.association import AssociationResult
from semantic_mapping.geometry_utils import exceeds_size_limit, fit_ground_plane, foreground_depth_mask, parse_size_limits
from semantic_mapping.object_map import ObjectMap
from semantic_mapping.persistence import load_map, save_map
from semantic_mapping.pipeline import PipelineConfig, SemanticMappingPipeline
from semantic_mapping.types import CameraIntrinsics, Detection2D, Observation, StampedPose

W, H = 160, 120
K = np.array([[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]])
INTRINSICS = CameraIntrinsics(fx=100.0, fy=100.0, cx=80.0, cy=60.0, width=W, height=H)
# Camera 1.5 m above the ground looking along world +x; world z is up.
R_WORLD_FROM_CAM = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _pose(x=0.0, y=0.0):
    T = np.eye(4)
    T[:3, :3] = R_WORLD_FROM_CAM
    T[:3, 3] = [x, y, 1.5]
    return T


def _render(boxes, slope=0.0, wall_x=40.0, T=None, ring_step=4, ring_offset=0):
    """Ray-cast ground ``z = slope * x``, world boxes, and a far wall; keep every
    ``ring_step``-th image row like a LiDAR ring pattern. Returns (depth, hit box index)."""
    T = _pose() if T is None else T
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    rays_cam = np.stack(((us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us, dtype=float)), -1)
    d = rays_cam @ T[:3, :3].T  # t along d is the camera depth
    o = T[:3, 3]
    depth = np.full((H, W), np.inf)
    hit = np.full((H, W), -1)
    denominator = d[..., 2] - slope * d[..., 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(denominator < 0, (slope * o[0] - o[2]) / denominator, np.inf)
        depth = np.minimum(depth, np.where(t > 0, t, np.inf))
        t = np.where(d[..., 0] > 0, (wall_x - o[0]) / d[..., 0], np.inf)
        depth = np.minimum(depth, t)
        for i, (lo, hi) in enumerate(boxes):
            t1, t2 = (np.asarray(lo) - o) / d, (np.asarray(hi) - o) / d
            near, far = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
            inside = (near <= far) & (near > 0) & (near < depth)
            depth = np.where(inside, near, depth)
            hit = np.where(inside, i, hit)
    depth[~np.isfinite(depth)] = 0.0
    sparse = np.zeros_like(depth)
    sparse[ring_offset::ring_step] = depth[ring_offset::ring_step]
    return sparse, hit


def _dense_mask(hit, index, grow_down=8, grow_up=0):
    """The object's silhouette, bled downward onto the ground (and optionally up)."""
    mask = hit == index
    grown = mask.copy()
    for k in range(1, grow_down + 1):
        grown[k:] |= mask[:-k]
    for k in range(1, grow_up + 1):
        grown[:-k] |= mask[k:]
    return grown


BAG = ([10.0, -0.5, 0.0], [11.0, 0.5, 1.0])


def _bag_detection(hit, label='bulk bag', **grow):
    mask = _dense_mask(hit, 0, **grow)
    ys, xs = np.nonzero(mask)
    return Detection2D(bbox=np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float),
                       label=label, score=0.9, mask=mask)


def _lift(config, detection, depth, T=None):
    return SemanticMappingPipeline(config)._detection_points_world(detection, depth, K, _pose() if T is None else T)


FOREGROUND = dict(foreground_depth_gap_m=0.75, foreground_depth_min_fraction=0.05,
                  foreground_depth_labels=['bulk bag'])


# ------------------------------------------------------------ A: ground removal
def test_fit_ground_plane_recovers_slope_under_objects_and_falls_back_level():
    rng = np.random.default_rng(0)
    xy = rng.uniform([5, -3], [15, 3], size=(400, 2))
    ground = np.column_stack((xy, 0.05 * xy[:, 0] + 0.3))
    body = np.column_stack((rng.uniform([9, -0.5], [10, 0.5], size=(200, 2)), rng.uniform(0.8, 2.0, 200)))
    body[:, 2] += 0.05 * body[:, 0]
    a, b, c = fit_ground_plane(np.vstack((ground, body)))
    assert abs(a - 0.05) < 0.01 and abs(b) < 0.01 and abs(c - 0.3) < 0.05
    level = fit_ground_plane(ground[:2])  # too few cells for a plane
    assert level[0] == level[1] == 0 and level[2] == pytest.approx(ground[:2, 2].min())
    assert fit_ground_plane(np.zeros((0, 3))) is None


@pytest.mark.parametrize('slope', [0.0, 0.04])
def test_ground_removal_keeps_the_object_not_the_ground_strip_in_front(slope):
    lo, hi = np.array(BAG[0]), np.array(BAG[1])
    lo[2] += slope * 10.0
    hi[2] += slope * 10.0
    depth, hit = _render([(lo, hi)], slope=slope)
    detection = _bag_detection(hit)
    # The nearest supported layer is a ground ring in front of the bag: a flat slab.
    slab = _lift(PipelineConfig(**FOREGROUND), detection, depth)
    assert len(slab) and np.ptp(slab[:, 2]) < 0.2 and slab[:, 0].max() < 10.0
    points = _lift(PipelineConfig(**FOREGROUND, ground_removal_labels=['bulk bag']), detection, depth)
    assert len(points)
    assert points[:, 0].min() >= 10.0 - 1e-6 and points[:, 0].max() <= 11.0 + 1e-6
    assert points[:, 2].min() >= lo[2] + 0.1 and points[:, 2].max() > hi[2] - 0.35  # rings every ~0.4 m
    # Scoped by label; other classes are untouched.
    other = _lift(PipelineConfig(ground_removal_labels=['person']), detection, depth)
    np.testing.assert_array_equal(other, _lift(PipelineConfig(), detection, depth))


def test_ground_removal_does_not_modify_the_frame_depth():
    depth, hit = _render([BAG])
    original = depth.copy()
    _lift(PipelineConfig(ground_removal_labels=['bulk bag']), _bag_detection(hit), depth)
    np.testing.assert_array_equal(depth, original)


# ---------------------------------------------------------------- B: layer choice
def test_largest_layer_selection_picks_the_object_over_nearer_ground_rings():
    depth, hit = _render([BAG])
    detection = _bag_detection(hit)
    points = _lift(PipelineConfig(**FOREGROUND, foreground_depth_largest_labels=['bulk bag']), detection, depth)
    # The object's layer (plus the ground ring touching it), not the nearest ring alone.
    assert points[:, 0].min() > 9.0 and points[:, 2].max() > 0.6
    depths = np.array([5.0] * 6 + [9.0] * 20 + [30.0] * 10)
    assert foreground_depth_mask(depths, 0.75, 5, 0.1).sum() == 6  # nearest (default)
    largest = foreground_depth_mask(depths, 0.75, 5, 0.1, select='largest')
    np.testing.assert_array_equal(depths[largest], 9.0)
    with pytest.raises(ValueError):
        foreground_depth_mask(depths, 0.75, select='median')


# ------------------------------------------------------- C: support-based bounds
def _cloud(center, n=60, spread=0.4, seed=0):
    return np.asarray(center) + np.random.default_rng(seed).uniform(-spread, spread, size=(n, 3))


def _match(object_map, instance, points, stamp):
    detection = Detection2D(bbox=np.array([0.0, 0.0, 10.0, 10.0]), label='bulk bag', score=0.9)
    object_map.update_matched(instance, instance.track, points, detection, stamp, K, np.eye(4), None)


def test_supported_bounds_drop_an_unconfirmed_early_outlier():
    body = _cloud([10.0, 0.0, 0.5])
    outlier = _cloud([10.0, 8.0, 0.0], n=20, spread=0.3, seed=1)  # ground/background caught once
    for min_support, expect_shrink in ((1, False), (3, True)):
        om = ObjectMap(bbox_min_support=min_support, bbox_support_radius_m=0.3)
        obj = om.spawn(np.array([0.0, 0.0, 10.0, 10.0]), np.vstack((body, outlier)), 'bulk bag', 0.9, 0.0)
        assert obj.bbox3d[4] > 7.0
        for i in range(1, 4):
            jitter = np.random.default_rng(10 + i).normal(0, 0.02, body.shape)
            _match(om, obj, body + jitter, i * 0.1)
        assert (obj.bbox3d[4] < 1.0) == expect_shrink
        assert len(obj.points_world) >= len(body)  # points are kept; only the reported box changes
    assert obj.point_support.shape == (len(obj.points_world),) and obj.point_support.max() >= 4


def test_young_instances_report_all_points_and_merges_keep_best_support(tmp_path):
    om = ObjectMap(bbox_min_support=3)
    points = _cloud([0.0, 0.0, 0.0], spread=0.2)
    a = om.spawn(np.array([0.0, 0.0, 10.0, 10.0]), points, 'bulk bag', 0.9, 0.0)
    b = om.spawn(np.array([0.0, 0.0, 10.0, 10.0]), points + [0.01, 0, 0], 'bulk bag', 0.9, 0.0)
    _match(om, b, points + [0.01, 0, 0], 0.1)
    _match(om, b, points + [0.01, 0, 0], 0.2)
    np.testing.assert_allclose(a.bbox3d, om._bbox(a.points_world))  # young: all points
    assert om.merge_duplicates() == [(a.instance_id, b.instance_id)]
    assert a.point_support.max() == 3 and a.point_support.shape == (len(a.points_world),)
    save_map(om, tmp_path)
    restored = ObjectMap(bbox_min_support=3)
    load_map(tmp_path, restored, resume=False)
    np.testing.assert_array_equal(restored.objects[a.instance_id].point_support, a.point_support)


def test_support_tracking_is_off_by_default():
    om = ObjectMap()
    obj = om.spawn(np.array([0.0, 0.0, 10.0, 10.0]), _cloud([0.0, 0.0, 0.0]), 'bulk bag', 0.9, 0.0)
    _match(om, obj, _cloud([0.0, 0.0, 0.0], seed=3), 0.1)
    assert obj.point_support.size == 0
    assert SemanticMappingPipeline(PipelineConfig(bbox_min_support=4)).object_map.bbox_min_support == 4


# ----------------------------------------------------------- D: class size limits
def test_size_limit_parsing_and_validation():
    limits = parse_size_limits(['concrete barrier:5', 'person: 1.2, 2.2'])
    assert limits == {'concrete barrier': (5.0,), 'person': (1.2, 2.2)}
    assert exceeds_size_limit(np.array([0, 0, 0, 3, 4, 1.0]), (5.0,))
    assert not exceeds_size_limit(np.array([0, 0, 0, 3, 4, 0.0]), (5.0,))
    assert exceeds_size_limit(np.array([0, 0, 0, 1, 0, 2.3]), (1.2, 2.2))
    for bad in (['nolimit'], [':3'], ['a:0'], ['a:1,2,3'], ['a:x'], ['a:1', 'a:2'], ['a:inf']):
        with pytest.raises(ValueError):
            parse_size_limits(bad)
        with pytest.raises(ValueError):
            PipelineConfig(class_size_limits=bad)
    for name in ('ground_removal_labels', 'foreground_depth_largest_labels', 'class_size_limits'):
        with pytest.raises(ValueError):
            PipelineConfig(**{name: 'person'})
    for bad in ({'ground_clearance_m': 0}, {'ground_max_slope': float('nan')}, {'ground_context_px': -1},
                {'bbox_min_support': 0}, {'bbox_support_radius_m': -1}):
        with pytest.raises(ValueError):
            PipelineConfig(**bad)


def test_oversized_observation_loses_geometry_but_keeps_its_detection():
    depth, hit = _render([BAG])
    detection = _bag_detection(hit, grow_up=30)  # a loose mask reaching the far wall
    config = PipelineConfig(class_size_limits=['bulk bag:3'])
    pipeline = SemanticMappingPipeline(config)
    assert np.ptp(_lift(PipelineConfig(), detection, depth)[:, 0]) > 20
    observation = Observation(stamp=0.0, pose=StampedPose(0.0, _pose()), intrinsics=INTRINSICS, depth=depth,
                              detections=[detection])
    result = pipeline.process_frame(observation)
    assert result.detection_instance_ids == [1] and len(result.objects[0].points_world) == 0
    assert pipeline.object_map.stats['size_rejected_observations'] == 1
    detection = _bag_detection(hit)  # the tight mask is within the limit
    assert len(pipeline._detection_points_world(detection, depth, K, _pose()))


def _row_of_bags(limits):
    """Bags A, B, C side by side, 0.1 m apart. Loose masks cover A, then A+B,
    then B+C: each observation is bag-sized, but fusing them all is not."""
    pipeline = SemanticMappingPipeline(PipelineConfig(class_size_limits=limits))
    row = [(np.add(BAG[0], [0, 1.1 * i, 0]), np.add(BAG[1], [0, 1.1 * i, 0])) for i in range(3)]
    depth, hit = _render(row)
    ids = []
    for i, bags in enumerate(((0,), (0, 1), (1, 2))):
        mask = np.isin(hit, bags)
        ys, xs = np.nonzero(mask)
        detection = Detection2D(bbox=np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float),
                                label='bulk bag', score=0.9, mask=mask)
        observation = Observation(stamp=i * 0.1, pose=StampedPose(i * 0.1, _pose()), intrinsics=INTRINSICS,
                                  depth=depth, detections=[detection])
        ids += pipeline.process_frame(observation).detection_instance_ids
    return pipeline, ids


def test_association_that_would_snowball_past_the_class_limit_is_refused():
    pipeline, ids = _row_of_bags([])
    assert ids == [1, 1, 1] and np.ptp(pipeline.object_map.objects[1].points_world[:, 1]) > 3.0
    pipeline, ids = _row_of_bags(['bulk bag:2.5'])
    assert ids == [1, 1, 2]
    assert pipeline.object_map.stats['size_rejected_observations'] == 0
    assert pipeline.object_map.stats['size_refused_associations'] >= 1
    for obj in pipeline.object_map.objects.values():
        assert not exceeds_size_limit(obj.bbox3d, (2.5,))


def test_refusal_returns_pairs_to_the_unmatched_pools_and_ignores_depthless_matches():
    pipeline = SemanticMappingPipeline(PipelineConfig(class_size_limits=['bulk bag:2']))
    obj = pipeline.object_map.spawn(np.array([0.0, 0.0, 10.0, 10.0]), _cloud([0, 0, 0]), 'bulk bag', 0.9, 0.0)
    detections = [Detection2D(bbox=np.zeros(4), label='bulk bag', score=0.9) for _ in range(2)]
    result = AssociationResult(matches=[(0, 0), (1, 1)])
    pipeline._refuse_oversized(result, [obj, obj], detections, [_cloud([3, 0, 0]), np.zeros((0, 3))])
    assert result.matches == [(1, 1)] and result.unmatched_tracks == [0] and result.unmatched_detections == [0]
    # A detection inside an already-oversized instance does not grow it.
    assert not pipeline.object_map.growth_exceeds_limit(obj, obj.bbox3d + [0.1, 0.1, 0.1, -0.1, -0.1, -0.1])


def test_merge_that_would_exceed_the_class_limit_is_refused():
    for limits, expected in (({}, 1), ({'concrete barrier': (3.0,)}, 2)):
        om = ObjectMap(size_limits=limits)
        om.spawn(np.zeros(4), _cloud([0.0, 0.0, 0.5], spread=1.0), 'concrete barrier', 0.9, 0.0)
        om.spawn(np.zeros(4), _cloud([1.0, 0.0, 0.5], spread=1.0, seed=2), 'concrete barrier', 0.9, 0.0)
        merged = om.merge_duplicates(iou_threshold=0.2)
        assert len(om.objects) == expected
        assert om.stats['size_refused_merges'] == (0 if merged else 1)


# ------------------------------------------------ F/G: mask-footprint completion
def _extent(points):
    return np.ptp(points, axis=0)


def test_completion_spans_the_mask_where_only_one_ring_hits_the_object():
    depth, hit = _render([BAG], ring_step=8)  # a single ring crosses the bag
    detection = _bag_detection(hit)
    base = dict(FOREGROUND, ground_removal_labels=['bulk bag'])
    sparse = _lift(PipelineConfig(**base), detection, depth)
    assert len(sparse) and _extent(sparse)[2] < 0.1  # a sliver on the one ring
    config = PipelineConfig(**base, mask_completion_labels=['bulk bag'])
    pipeline = SemanticMappingPipeline(config)
    points = pipeline._detection_points_world(detection, depth, K, _pose())
    dx, dy, dz = _extent(points)
    assert abs(dy - 1.0) < 0.2 and abs(dz - 1.0) < 0.25 and dx < 0.2
    assert points[:, 0].min() >= 9.9 and points[:, 2].min() >= -1e-6  # the bled ground strip is not completed
    assert pipeline.object_map.stats['mask_completions'] == 1
    assert len(points) <= config.max_points_per_detection
    capped = PipelineConfig(**base, mask_completion_labels=['bulk bag'], max_points_per_detection=len(sparse) + 4)
    assert len(_lift(capped, detection, depth)) == len(sparse) + 4
    # Too few kept returns: no completion.
    few = PipelineConfig(**base, mask_completion_labels=['bulk bag'], mask_completion_min_returns=1000)
    np.testing.assert_array_equal(_lift(few, detection, depth), sparse)


def test_completion_is_off_by_default_and_leaves_the_frame_depth_alone():
    depth, hit = _render([BAG], ring_step=8)
    original = depth.copy()
    detection = _bag_detection(hit)
    default = _lift(PipelineConfig(**FOREGROUND), detection, depth)
    other = _lift(PipelineConfig(**FOREGROUND, mask_completion_labels=['person'],
                                 ground_contact_depth_labels=['person']), detection, depth)
    np.testing.assert_array_equal(default, other)
    _lift(PipelineConfig(mask_completion_labels=['bulk bag']), detection, depth)
    np.testing.assert_array_equal(depth, original)


@pytest.mark.parametrize('slope', [0.0, 0.03])
def test_ground_contact_recovers_range_without_returns_on_the_object(slope):
    lo, hi = np.array(BAG[0]), np.array(BAG[1])
    lo[2] += slope * 10.0
    hi[2] += slope * 10.0
    depth, hit = _render([(lo, hi)], slope=slope, ring_step=16, ring_offset=12)
    detection = _bag_detection(hit, grow_down=0)
    assert not (depth[detection.mask] > 0).any()
    base = dict(ground_context_px=40, mask_completion_labels=['bulk bag'])
    assert len(_lift(PipelineConfig(**base), detection, depth)) == 0
    pipeline = SemanticMappingPipeline(PipelineConfig(**base, ground_contact_depth_labels=['bulk bag']))
    points = pipeline._detection_points_world(detection, depth, K, _pose())
    assert pipeline.object_map.stats['ground_contact_completions'] == 1
    assert abs(np.median(points[:, 0]) - 10.0) < 0.5
    assert abs(_extent(points)[1] - 1.0) < 0.25 and abs(_extent(points)[2] - 1.0) < 0.3
    # Beyond the usable depth range: no geometry.
    far = SemanticMappingPipeline(PipelineConfig(**base, ground_contact_depth_labels=['bulk bag'], max_depth_m=8.0))
    assert len(far._detection_points_world(detection, depth, K, _pose())) == 0


def test_ground_contact_needs_a_fitted_ground():
    depth, hit = _render([BAG], ring_step=16, ring_offset=12)
    depth[hit < 0] = 0.0  # nothing around the object to fit the ground to
    config = PipelineConfig(ground_contact_depth_labels=['bulk bag'])
    assert len(_lift(config, _bag_detection(hit, grow_down=0), depth)) == 0


def test_size_limit_still_rejects_a_completed_mask_that_covers_background():
    depth, hit = _render([BAG], ring_step=8)
    detection = _bag_detection(hit)
    detection.mask[20:hit.shape[0] // 2, :] = True  # bled across the sky/far wall rows
    base = dict(FOREGROUND, ground_removal_labels=['bulk bag'], mask_completion_labels=['bulk bag'])
    assert _extent(_lift(PipelineConfig(**base), detection, depth))[1] > 5
    pipeline = SemanticMappingPipeline(PipelineConfig(**base, class_size_limits=['bulk bag:2.5']))
    assert len(pipeline._detection_points_world(detection, depth, K, _pose())) == 0
    assert pipeline.object_map.stats['size_rejected_observations'] == 1


def test_completed_points_accumulate_support_like_any_other_frame():
    depth, hit = _render([BAG], ring_step=8)
    config = PipelineConfig(**FOREGROUND, ground_removal_labels=['bulk bag'], mask_completion_labels=['bulk bag'],
                            bbox_min_support=3, bbox_support_radius_m=0.3)
    pipeline = SemanticMappingPipeline(config)
    for i, x in enumerate((0.0, 0.1, -0.1, 0.05)):  # small range changes between frames
        observation = Observation(stamp=i * 0.1, pose=StampedPose(i * 0.1, _pose(x=x)), intrinsics=INTRINSICS,
                                  depth=_render([BAG], ring_step=8, T=_pose(x=x))[0],
                                  detections=[_bag_detection(_render([BAG], T=_pose(x=x))[1])])
        result = pipeline.process_frame(observation)
    obj = result.objects[0]
    assert len(result.objects) == 1 and obj.point_support.max() >= 3
    dx, dy, dz = obj.bbox3d[3:] - obj.bbox3d[:3]
    assert abs(dy - 1.0) < 0.3 and abs(dz - 1.0) < 0.35
