import numpy as np

from semantic_mapping import association
from semantic_mapping.tracking import init_track


def test_associate_matches_by_best_iou():
    track_a = init_track(np.array([0.0, 0.0, 10.0, 10.0]))
    track_b = init_track(np.array([100.0, 100.0, 110.0, 110.0]))
    predicted_bboxes = [np.array([0.0, 0.0, 10.0, 10.0]), np.array([100.0, 100.0, 110.0, 110.0])]
    detections = [np.array([101.0, 101.0, 111.0, 111.0]), np.array([1.0, 1.0, 11.0, 11.0])]

    result = association.associate([track_a, track_b], predicted_bboxes, detections, iou_threshold=0.3)

    assert (0, 1) in result.matches  # track_a <-> detections[1]
    assert (1, 0) in result.matches  # track_b <-> detections[0]
    assert result.unmatched_tracks == []
    assert result.unmatched_detections == []


def test_associate_leaves_low_iou_pair_unmatched():
    track = init_track(np.array([0.0, 0.0, 10.0, 10.0]))
    predicted_bboxes = [np.array([0.0, 0.0, 10.0, 10.0])]
    detections = [np.array([500.0, 500.0, 510.0, 510.0])]

    result = association.associate([track], predicted_bboxes, detections, iou_threshold=0.3)

    assert result.matches == []
    assert result.unmatched_tracks == [0]
    assert result.unmatched_detections == [0]


def test_associate_empty_inputs():
    result = association.associate([], [], [np.array([0.0, 0.0, 1.0, 1.0])])
    assert result.matches == []
    assert result.unmatched_detections == [0]

    track = init_track(np.array([0.0, 0.0, 10.0, 10.0]))
    result2 = association.associate([track], [np.array([0.0, 0.0, 10.0, 10.0])], [])
    assert result2.unmatched_tracks == [0]


def test_should_reactivate_window():
    assert association.should_reactivate(frames_since_seen=5, max_occlusion_frames=10) is True
    assert association.should_reactivate(frames_since_seen=11, max_occlusion_frames=10) is False


def _loop_cost(predicted_bboxes, detection_bboxes, rows, cols, iou_threshold, beliefs, labels, min_mass, weight):
    """The per-pair cost loop associate() ran before its cost matrix was vectorised."""
    from semantic_mapping.geometry_utils import iou_xyxy

    cost = np.full((len(rows), len(cols)), association.INVALID_COST)
    for r, i in enumerate(rows):
        for c, j in enumerate(cols):
            label_cost = 0.0
            if beliefs is not None and labels is not None:
                mass = association.label_mass(labels[j], beliefs[i])
                if not association.labels_compatible(labels[j], beliefs[i], min_mass):
                    continue
                label_cost = weight * (1.0 - mass)
            iou = iou_xyxy(predicted_bboxes[i], detection_bboxes[j])
            if iou >= iou_threshold:
                cost[r, c] = 1.0 - iou + label_cost
    return cost


def test_associate_cost_matrix_matches_the_pairwise_loop(monkeypatch):
    captured = []
    solve = association._solve
    monkeypatch.setattr(association, "_solve", lambda cost, *rest: captured.append(cost.copy()) or solve(cost, *rest))
    rng = np.random.default_rng(11)
    names = ["chair", "table", "box", "lamp"]
    for trial in range(60):
        n, m = rng.integers(0, 15, size=2)

        def box():
            x, y = rng.integers(0, 30, size=2) * 4.0
            w, h = rng.integers(1, 12, size=2) * 4.0
            b = np.array([x, y, x + w, y + h])
            return b + rng.normal(0.0, 0.5, 4) if trial % 2 else b

        predicted = [box() for _ in range(n)]
        detections = [box() for _ in range(m)]
        tracks = [init_track(b) for b in predicted]
        beliefs = [{name: float(p) for name, p in zip(names, rng.dirichlet(np.ones(4))) if p > 0.05}
                   if rng.random() < 0.5 else {names[int(rng.integers(4))]: 0.2, "person": 0.8}
                   for _ in range(n)]
        labels = [names[int(rng.integers(4))] for _ in range(m)]
        rows = sorted(rng.choice(n, size=int(rng.integers(0, n + 1)), replace=False).tolist()) if trial % 3 else None
        cols = sorted(rng.choice(m, size=int(rng.integers(0, m + 1)), replace=False).tolist()) if trial % 3 else None
        # Half-shifted lattice boxes have IoU exactly 1/3; masses of exactly 0.2 tie label_min_mass.
        for label_aware, threshold in ((False, 0.25), (False, 1 / 3), (True, 0.25), (True, 1 / 3)):
            kwargs = dict(track_label_beliefs=beliefs, detection_labels=labels, label_min_mass=0.2,
                          label_cost_weight=0.1) if label_aware else {}
            captured.clear()
            association.associate(tracks, predicted, detections, iou_threshold=threshold, candidate_tracks=rows,
                                  candidate_detections=cols, use_mahalanobis_gate=False, **kwargs)
            expected = _loop_cost(predicted, detections, list(range(n)) if rows is None else rows,
                                  list(range(m)) if cols is None else cols, threshold,
                                  beliefs if label_aware else None, labels if label_aware else None, 0.2, 0.1)
            assert np.array_equal(captured[0], expected)
