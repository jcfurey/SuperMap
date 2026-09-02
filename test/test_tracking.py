import numpy as np

from semantic_mapping import tracking


def test_bbox_measurement_round_trip():
    bbox = np.array([10.0, 20.0, 30.0, 60.0])
    measurement = tracking.bbox_to_measurement(bbox)
    assert np.allclose(measurement, [20.0, 40.0, 20.0, 40.0])
    assert np.allclose(tracking.measurement_to_bbox(measurement), bbox)


def test_init_track_matches_first_detection():
    bbox = np.array([0.0, 0.0, 10.0, 20.0])
    track = tracking.init_track(bbox)
    assert np.allclose(tracking.current_bbox(track), bbox, atol=1e-6)
    assert np.allclose(tracking.current_velocity(track), [0.0, 0.0])


def test_update_moves_state_toward_measurement():
    track = tracking.init_track(np.array([0.0, 0.0, 10.0, 10.0]))
    measurement_bbox = np.array([20.0, 20.0, 30.0, 30.0])
    updated = tracking.update(track, measurement_bbox)
    # Should move substantially toward the new measurement, not stay put.
    assert updated.state[0] > track.state[0]
    assert updated.state[1] > track.state[1]


def test_predict_uses_projection_prior_over_linear_motion():
    track = tracking.init_track(np.array([0.0, 0.0, 10.0, 10.0]))
    # Give the track some velocity so a pure linear-motion prediction would
    # move it away from the projected pixel.
    track.state[4] = 100.0
    track.state[5] = 100.0

    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    T_world_from_cam = np.eye(4)
    object_centroid_world = np.array([0.0, 0.0, 2.0])  # projects to (50, 40)

    predicted = tracking.predict(
        track, dt=0.1, K=K, T_world_from_cam=T_world_from_cam,
        object_centroid_world=object_centroid_world, projection_prior_weight=1.0,
    )
    assert np.allclose(predicted.state[:2], [50.0, 40.0], atol=1e-6)


def test_predict_without_pose_falls_back_to_linear_motion():
    track = tracking.init_track(np.array([0.0, 0.0, 10.0, 10.0]))
    track.state[4] = 10.0  # vx
    track.state[5] = 0.0
    predicted = tracking.predict(track, dt=1.0)
    assert np.isclose(predicted.state[0], track.state[0] + 10.0)


def test_mahalanobis_gate_accepts_close_and_rejects_far():
    track = tracking.init_track(np.array([0.0, 0.0, 10.0, 10.0]))
    close_measurement = np.array([1.0, 1.0, 11.0, 11.0])
    far_measurement = np.array([500.0, 500.0, 510.0, 510.0])
    assert tracking.mahalanobis_gate(track, close_measurement) is True
    assert tracking.mahalanobis_gate(track, far_measurement) is False
