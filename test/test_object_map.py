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


def _plane_instance(object_map, instance_id, label, x_center, depth_m):
    """A 0.4 m square of points facing the camera at z = depth_m (camera at the origin looking +z)."""
    xs, ys = np.meshgrid(np.linspace(-0.2, 0.2, 9), np.linspace(-0.2, 0.2, 9))
    points = np.stack([xs.ravel() + x_center, ys.ravel(), np.full(xs.size, depth_m)], axis=1)
    obj = object_map.spawn(np.array([0.0, 0.0, 10.0, 10.0]), points, label, 0.9, stamp=0.0)
    obj.status = ObjectStatus.ACTIVE
    assert obj.instance_id == instance_id
    return obj


def test_out_of_view_instances_get_no_evidence_and_culling_is_equivalent():
    import copy

    K = np.array([[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]])
    depth = np.full((120, 160), 8.0)  # surface far behind every object: free-space evidence for what is in view
    results = {}
    for cull in (True, False):
        object_map = ObjectMap(cull_out_of_view=cull)
        in_view = _plane_instance(object_map, 1, "box", 0.0, 2.0)
        far_right = _plane_instance(object_map, 2, "box", 5.0, 2.0)   # projects far outside the 160 px image
        behind = _plane_instance(object_map, 3, "box", 0.0, -2.0)     # behind the camera
        for obj in (in_view, far_right, behind):
            object_map.update_unmatched(obj, K, np.eye(4), depth)
        results[cull] = copy.deepcopy(object_map.objects)

    for instance_id in (1, 2, 3):
        a, b = results[True][instance_id], results[False][instance_id]
        np.testing.assert_array_equal(a.point_log_odds, b.point_log_odds)
        assert a.status == b.status and a.points_contradicted == b.points_contradicted
    assert np.all(results[True][1].point_log_odds < 0)          # in view: contradicted by the far surface
    assert np.all(results[True][2].point_log_odds == 0)         # out of view: untouched
    assert np.all(results[True][3].point_log_odds == 0)


def test_may_be_in_view_keeps_boxes_straddling_the_camera_plane():
    K = np.array([[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]])
    assert ObjectMap.may_be_in_view(np.array([-0.5, -0.5, 1.0, 0.5, 0.5, 2.0]), K, np.eye(4), (120, 160))
    assert not ObjectMap.may_be_in_view(np.array([5.0, -0.5, 1.0, 6.0, 0.5, 2.0]), K, np.eye(4), (120, 160))
    assert not ObjectMap.may_be_in_view(np.array([-0.5, -0.5, -3.0, 0.5, 0.5, -2.0]), K, np.eye(4), (120, 160))
    assert ObjectMap.may_be_in_view(np.array([-0.5, -0.5, -1.0, 0.5, 0.5, 1.0]), K, np.eye(4), (120, 160))


def test_merge_duplicates_kdtree_candidates_match_exhaustive_pairs():
    object_map = ObjectMap()
    rng = np.random.default_rng(1)
    expected_drops = []
    next_id = 1
    for gx in range(6):
        for gy in range(5):
            base = np.array([3.0 * gx, 3.0 * gy, 0.0])
            points = base + rng.uniform(0.0, 0.5, size=(30, 3))
            obj = object_map.spawn(np.array([0, 0, 10, 10.0]), points, "crate", 0.9, 0.0)
            obj.status = ObjectStatus.ACTIVE
            next_id += 1
    live = sorted(object_map.objects.values(), key=lambda o: o.instance_id)
    for keep in live[:5]:  # five same-label duplicates, slightly offset: must merge into the older ID
        dup = object_map.spawn(np.array([0, 0, 10, 10.0]), keep.points_world + 0.03, "crate", 0.9, 1.0)
        dup.status = ObjectStatus.ACTIVE
        expected_drops.append((keep.instance_id, dup.instance_id))
    for keep in live[5:8]:  # three overlapping instances with a different label: never merged
        other = object_map.spawn(np.array([0, 0, 10, 10.0]), keep.points_world + 0.03, "barrel", 0.9, 1.0)
        other.status = ObjectStatus.ACTIVE

    merged = object_map.merge_duplicates(iou_threshold=0.3, distance_threshold=0.25)
    assert sorted(merged) == sorted(expected_drops)
    assert len(object_map.objects) == 30 + 3
