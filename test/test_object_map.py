import numpy as np

from semantic_mapping.object_map import ObjectMap, voxel_downsample
from semantic_mapping.types import ObjectStatus


def test_voxel_downsample_deduplicates_nearby_points():
    points = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [1.0, 1.0, 1.0]])
    downsampled = voxel_downsample(points, voxel_size=0.05)
    assert downsampled.shape[0] == 2


def test_spawn_creates_tentative_instance():
    m = ObjectMap()
    points = np.array([[0.0, 0.0, 1.0], [0.1, 0.1, 1.0]])
    obj = m.spawn(np.array([0.0, 0.0, 10.0, 10.0]), points, "chair", 0.9, stamp=0.0)
    assert obj.status == ObjectStatus.TENTATIVE
    assert obj.label == "chair"
    assert 1 in m.objects


def _identity_camera_looking_at_z():
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    T_world_from_cam = np.eye(4)
    depth = np.full((80, 100), 2.0)
    return K, T_world_from_cam, depth


def test_confirm_tentative_promotes_after_enough_hits():
    m = ObjectMap(min_observations_for_confidence_check=100)
    points = np.array([[0.0, 0.0, 2.0]])
    obj = m.spawn(np.array([40.0, 30.0, 60.0, 50.0]), points, "chair", 0.95, stamp=0.0)
    obj.hits = 2
    m.confirm_tentative(obj, min_hits=2)
    assert obj.status == ObjectStatus.ACTIVE


def test_confirm_tentative_discards_low_confidence_after_enough_observations():
    m = ObjectMap(min_label_confidence=0.9, min_observations_for_confidence_check=3)
    points = np.array([[0.0, 0.0, 2.0]])
    obj = m.spawn(np.array([40.0, 30.0, 60.0, 50.0]), points, "chair", 0.2, stamp=0.0)
    obj.hits = 5
    m.confirm_tentative(obj, min_hits=1)
    assert obj.status == ObjectStatus.DISAPPEARED


def test_update_unmatched_marks_disappeared_when_evidence_contradicts():
    m = ObjectMap(tau_eps=0.1, disappeared_occupied_fraction=0.2)
    K, T, _ = _identity_camera_looking_at_z()
    points = np.array([[0.0, 0.0, 2.0]])
    obj = m.spawn(np.array([40.0, 30.0, 60.0, 50.0]), points, "chair", 0.9, stamp=0.0)
    obj.point_log_odds = np.array([1.0])  # a single prior confirmation

    # A new depth reading much farther away means "nothing there anymore". The
    # log-odds filter should require sustained contradicting evidence (not a
    # single frame) before flipping a previously-confirmed point to disappeared.
    contradicting_depth = np.full((80, 100), 6.0)
    for _ in range(5):
        m.update_unmatched(obj, K, T, contradicting_depth)
    assert obj.status == ObjectStatus.DISAPPEARED


def test_update_unmatched_stays_active_when_still_geometrically_consistent():
    m = ObjectMap(tau_eps=0.1, active_occupied_fraction=0.5)
    K, T, depth = _identity_camera_looking_at_z()
    points = np.array([[0.0, 0.0, 2.0]])
    obj = m.spawn(np.array([40.0, 30.0, 60.0, 50.0]), points, "chair", 0.9, stamp=0.0)
    obj.point_log_odds = np.array([3.0])

    m.update_unmatched(obj, K, T, depth)  # depth still matches the point
    assert obj.status != ObjectStatus.DISAPPEARED


def test_prune_disappeared_removes_after_grace_period():
    m = ObjectMap()
    points = np.array([[0.0, 0.0, 2.0]])
    obj = m.spawn(np.array([40.0, 30.0, 60.0, 50.0]), points, "chair", 0.9, stamp=0.0)
    obj.status = ObjectStatus.DISAPPEARED
    obj.frames_since_seen = 100
    removed = m.prune_disappeared(grace_period_frames=10)
    assert removed == [obj.instance_id]
    assert obj.instance_id not in m.objects
