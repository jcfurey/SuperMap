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
