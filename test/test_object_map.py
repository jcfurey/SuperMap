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


def test_voxel_downsample_indices_keeps_first_occurrence_ascending():
    from semantic_mapping.object_map import voxel_downsample_indices

    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.001, 0.0, 0.0]])
    keep = voxel_downsample_indices(points, voxel_size=0.05)
    assert keep.tolist() == [0, 1]


def test_fuse_points_keeps_log_odds_aligned_with_points():
    m = ObjectMap(voxel_size=0.05)
    obj = m.spawn(np.array([0.0, 0.0, 10.0, 10.0]), np.array([[0.0, 0.0, 1.0]]), "chair", 0.9, stamp=0.0)
    obj.point_log_odds[:] = 4.0  # the existing point has accumulated evidence
    m._fuse_points(obj, np.array([[0.001, 0.0, 1.0], [2.0, 2.0, 2.0]]))  # one duplicate voxel, one new
    assert obj.points_world.shape[0] == 2
    assert obj.point_log_odds.tolist() == [4.0, 0.0]  # existing evidence kept, new point neutral


def test_merge_duplicates_keeps_older_id_and_unions_points():
    m = ObjectMap()
    older = m.spawn(np.array([0.0, 0.0, 10.0, 10.0]), np.array([[0.0, 0.0, 1.0], [0.5, 0.5, 1.5]]), "chair", 0.9, 0.0)
    newer = m.spawn(np.array([0.0, 0.0, 10.0, 10.0]), np.array([[0.1, 0.1, 1.0], [0.6, 0.6, 1.6]]), "chair", 0.9, 0.1)
    older.status = ObjectStatus.OCCLUDED
    newer.status = ObjectStatus.ACTIVE
    merged = m.merge_duplicates(iou_threshold=0.3, distance_threshold=0.25)
    assert merged == [(older.instance_id, newer.instance_id)]
    assert newer.instance_id not in m.objects
    assert older.points_world.shape[0] == 4
    assert older.status == ObjectStatus.ACTIVE  # takes the stronger status
    assert older.hits == 2


def test_merge_duplicates_ignores_different_labels_and_far_objects():
    m = ObjectMap()
    m.spawn(np.array([0.0, 0.0, 10.0, 10.0]), np.array([[0.0, 0.0, 1.0], [0.5, 0.5, 1.5]]), "chair", 0.9, 0.0)
    m.spawn(np.array([0.0, 0.0, 10.0, 10.0]), np.array([[0.0, 0.0, 1.0], [0.5, 0.5, 1.5]]), "table", 0.9, 0.0)
    m.spawn(np.array([0.0, 0.0, 10.0, 10.0]), np.array([[9.0, 9.0, 1.0], [9.5, 9.5, 1.5]]), "chair", 0.9, 0.0)
    assert m.merge_duplicates() == []
    assert len(m.objects) == 3


def test_tentative_instance_expires_when_never_corroborated():
    m = ObjectMap(tentative_max_age=2)
    K, T, depth = _identity_camera_looking_at_z()
    obj = m.spawn(np.array([40.0, 30.0, 60.0, 50.0]), np.array([[0.0, 0.0, 2.0]]), "chair", 0.9, stamp=0.0)
    for _ in range(2):
        m.update_unmatched(obj, K, T, depth)
        assert obj.status == ObjectStatus.TENTATIVE  # within its grace window, evidence still consistent
    m.update_unmatched(obj, K, T, depth)
    assert obj.status == ObjectStatus.DISAPPEARED


def test_point_outside_detected_region_is_pruned_by_membership():
    from semantic_mapping.types import Detection2D
    from semantic_mapping.tracking import init_track

    m = ObjectMap(tau_eps=0.1, prune_membership=-1.5)
    K, T, _ = _identity_camera_looking_at_z()
    depth = np.full((80, 100), 2.0)  # everything at 2 m: both points are geometrically "real"
    # One point projects to (50, 40) inside the box, the other to (90, 40) well outside it.
    points = np.array([[0.0, 0.0, 2.0], [0.8, 0.0, 2.0]])
    obj = m.spawn(np.array([40.0, 30.0, 60.0, 50.0]), points, "chair", 0.9, stamp=0.0)
    det = Detection2D(bbox=np.array([40.0, 30.0, 60.0, 50.0]), label="chair", score=0.9)

    for i in range(3):
        m.update_matched(obj, init_track(det.bbox), np.zeros((0, 3)), det, 0.1 * (i + 1), K, T, depth)

    assert obj.points_world.shape[0] == 1
    assert np.allclose(obj.points_world[0], [0.0, 0.0, 2.0])
    assert obj.point_membership[0] > 0
    assert obj.point_log_odds[0] > 0  # the surviving point is also geometrically confirmed


def test_mask_detection_uses_mask_not_box_for_membership():
    from semantic_mapping.object_map import _inside_detection
    from semantic_mapping.types import Detection2D

    mask = np.zeros((80, 100), dtype=bool)
    mask[35:45, 45:55] = True
    det = Detection2D(bbox=np.array([0.0, 0.0, 100.0, 80.0]), label="chair", score=0.9, mask=mask)
    pixels = np.array([[50, 40], [90, 40], [-1, -1]])
    assert _inside_detection(pixels, det, margin_px=2.0).tolist() == [True, False, False]


def test_disappearance_counts_contradicted_points_even_after_pruning():
    """Most of an object's points are contradicted (and pruned); a few floor-
    contact points stay geometrically consistent. Survivors alone would look
    fully occupied -- the contradiction count must still flag disappearance."""
    m = ObjectMap(tau_eps=0.1, disappeared_occupied_fraction=0.2, prune_log_odds=-1.5)
    K, T, _ = _identity_camera_looking_at_z()
    # 9 points at z = 2 (will be contradicted by a farther reading), 1 at z = 5 (stays consistent).
    points = np.concatenate([np.tile([[0.0, 0.0, 2.0]], (9, 1)) + np.arange(9)[:, None] * [0.05, 0.0, 0.0],
                             [[0.0, 0.5, 5.0]]])
    obj = m.spawn(np.array([40.0, 30.0, 60.0, 50.0]), points, "chair", 0.9, stamp=0.0)
    obj.status = ObjectStatus.ACTIVE
    depth = np.full((80, 100), 5.0)
    for _ in range(4):
        m.update_unmatched(obj, K, T, depth)
    assert obj.points_contradicted >= 9
    assert obj.status == ObjectStatus.DISAPPEARED


def test_bbox_is_padded_by_half_voxel_so_single_face_views_have_volume():
    m = ObjectMap(voxel_size=0.1)
    obj = m.spawn(np.array([0.0, 0.0, 10.0, 10.0]), np.array([[0.0, 0.0, 1.0], [0.0, 0.5, 1.5]]), "chair", 0.9, 0.0)
    dims = obj.bbox3d[3:] - obj.bbox3d[:3]
    assert np.allclose(dims, [0.1, 0.6, 0.6])
