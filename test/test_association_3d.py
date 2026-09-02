import numpy as np

from semantic_mapping import association
from semantic_mapping.tracking import init_track
from test.helpers import make_object


def _obj(instance_id, label, bbox):
    obj = make_object(instance_id, label, bbox)
    obj.points_world = np.array([[bbox[0], bbox[1], bbox[2]]])
    return obj


def test_associate_3d_matches_overlapping_box_with_compatible_label():
    chair = _obj(1, "chair", (0.0, 0.0, 0.0, 0.5, 0.5, 0.9))
    det_box = np.array([0.1, 0.1, 0.0, 0.6, 0.6, 0.9])
    result = association.associate_3d([det_box], ["chair"], [chair])
    assert result.matches == [(0, 0)]


def test_associate_3d_rejects_incompatible_label():
    chair = _obj(1, "chair", (0.0, 0.0, 0.0, 0.5, 0.5, 0.9))
    det_box = np.array([0.1, 0.1, 0.0, 0.6, 0.6, 0.9])
    result = association.associate_3d([det_box], ["table"], [chair])
    assert result.matches == []
    assert result.unmatched_detections == [0]


def test_associate_3d_containment_rescues_small_partial_view_of_large_object():
    sofa = _obj(1, "sofa", (-1.0, -0.5, 0.0, 1.0, 0.5, 0.8))
    corner = np.array([0.7, 0.3, 0.0, 1.05, 0.55, 0.4])  # tiny IoU, but centroid inside sofa (+margin)
    result = association.associate_3d([corner], ["sofa"], [sofa], iou_threshold=0.5, containment_margin=0.1)
    assert result.matches == [(0, 0)]


def test_associate_3d_ignores_detections_without_box():
    chair = _obj(1, "chair", (0.0, 0.0, 0.0, 0.5, 0.5, 0.9))
    result = association.associate_3d([None], ["chair"], [chair])
    assert result.matches == []


def test_associate_3d_respects_candidate_subsets_and_returns_original_indices():
    a = _obj(1, "chair", (0.0, 0.0, 0.0, 0.5, 0.5, 0.9))
    b = _obj(2, "chair", (5.0, 5.0, 0.0, 5.5, 5.5, 0.9))
    det_far = np.array([9.0, 9.0, 0.0, 9.5, 9.5, 0.9])
    det_b = np.array([5.1, 5.1, 0.0, 5.6, 5.6, 0.9])
    result = association.associate_3d(
        [det_far, det_b], ["chair", "chair"], [a, b], candidate_objects=[1], candidate_detections=[1],
    )
    assert result.matches == [(1, 1)]
    assert result.unmatched_tracks == []
    assert result.unmatched_detections == []


def test_associate_2d_candidate_subsets():
    tracks = [init_track(np.array([0.0, 0.0, 10.0, 10.0])), init_track(np.array([50.0, 50.0, 60.0, 60.0]))]
    predicted = [np.array([0.0, 0.0, 10.0, 10.0]), np.array([50.0, 50.0, 60.0, 60.0])]
    dets = [np.array([1.0, 1.0, 11.0, 11.0]), np.array([51.0, 51.0, 61.0, 61.0])]
    result = association.associate(tracks, predicted, dets, candidate_tracks=[1], candidate_detections=[1])
    assert result.matches == [(1, 1)]
    assert result.unmatched_tracks == [] and result.unmatched_detections == []
