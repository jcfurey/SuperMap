import numpy as np
import pytest

from semantic_mapping import geometric_consistency as gc


def _identity_camera():
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    T_world_from_cam = np.eye(4)
    return K, T_world_from_cam


def test_classify_points_observable_when_depth_matches():
    K, T = _identity_camera()
    depth_image = np.full((80, 100), 2.0)
    points_world = np.array([[0.0, 0.0, 2.0]])  # projects to (50, 40), matches depth
    states, delta_d = gc.classify_points(K, T, depth_image, points_world, tau_eps=0.1)
    assert states[0] == gc.GeometricState.OBSERVABLE
    assert np.isclose(delta_d[0], 0.0)


def test_classify_points_disappeared_when_point_in_front_of_surface():
    K, T = _identity_camera()
    depth_image = np.full((80, 100), 5.0)  # sensor now sees something farther away
    points_world = np.array([[0.0, 0.0, 2.0]])  # old point much closer than new surface
    states, _ = gc.classify_points(K, T, depth_image, points_world, tau_eps=0.1)
    assert states[0] == gc.GeometricState.DISAPPEARED


def test_classify_points_unobservable_when_point_behind_surface():
    K, T = _identity_camera()
    depth_image = np.full((80, 100), 1.0)  # something closer is occluding the point
    points_world = np.array([[0.0, 0.0, 5.0]])
    states, _ = gc.classify_points(K, T, depth_image, points_world, tau_eps=0.1)
    assert states[0] == gc.GeometricState.UNOBSERVABLE


def test_classify_points_out_of_view_when_behind_camera():
    K, T = _identity_camera()
    depth_image = np.full((80, 100), 2.0)
    points_world = np.array([[0.0, 0.0, -1.0]])  # behind the camera
    states, delta_d = gc.classify_points(K, T, depth_image, points_world, tau_eps=0.1)
    assert states[0] == gc.GeometricState.OUT_OF_VIEW
    assert np.isnan(delta_d[0])


def test_vectorized_state_comparison_matches_scalar_comparison():
    # Regression test: numpy's == against a str-mixin Enum scalar was
    # silently returning all-False for a vectorized comparison even when
    # individual scalar comparisons returned True (see geometric_consistency
    # module docstring on GeometricState). Guards against reintroducing an
    # object-dtype/str-Enum state array.
    states = np.array([
        gc.GeometricState.OBSERVABLE, gc.GeometricState.DISAPPEARED, gc.GeometricState.OBSERVABLE,
    ])
    mask = states == gc.GeometricState.OBSERVABLE
    assert mask.tolist() == [True, False, True]


def test_inverse_sensor_model_maps_states_to_probabilities():
    states = np.array([gc.GeometricState.OBSERVABLE, gc.GeometricState.DISAPPEARED, gc.GeometricState.UNOBSERVABLE])
    probs = gc.inverse_sensor_model(states, p_hit=0.9, p_miss=0.1, p_unknown=0.5)
    assert np.allclose(probs, [0.9, 0.1, 0.5])


def test_log_odds_update_is_monotonic_and_clamped():
    prior = np.array([0.0])
    after_hit = gc.update_log_odds(prior, np.array([0.9]))
    assert after_hit[0] > 0.0

    # Repeated strong hits should clamp, not diverge to infinity.
    saturated = prior.copy()
    for _ in range(100):
        saturated = gc.update_log_odds(saturated, np.array([0.999]))
    assert saturated[0] <= gc.LOG_ODDS_MAX


def test_occupied_fraction_and_prune_mask():
    log_odds = np.array([5.0, 5.0, -5.0])
    assert gc.occupied_fraction(log_odds) == pytest.approx(2 / 3)
    assert gc.prune_mask(log_odds, prune_threshold=-1.5).tolist() == [False, False, True]


def test_classifying_many_point_sets_at_once_matches_one_call_per_set():
    rng = np.random.default_rng(4)
    K = np.array([[120.0, 0.0, 64.5], [0.0, 120.0, 48.5], [0.0, 0.0, 1.0]])
    for trial in range(20):
        angle = rng.uniform(-0.5, 0.5)
        T = np.eye(4)
        T[:3, :3] = [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
        T[:3, 3] = rng.normal(0.0, 0.3, 3)
        depth = rng.uniform(1.0, 4.0, size=(96, 128))
        depth[rng.random(depth.shape) < 0.2] = 0.0         # missing readings
        depth[rng.random(depth.shape) < 0.05] = np.nan
        # Some sets straddle the image border or lie behind the camera; one is empty.
        sets = [rng.normal([0.0, 0.0, 2.5], [1.5, 1.0, 1.5], size=(int(rng.integers(0, 400)), 3))
                for _ in range(int(rng.integers(1, 8)))]
        for window in (0, 2):
            together = gc.project_and_classify_many(K, T, depth, sets, 0.1, window)
            assert len(together) == len(sets)
            for points, batched in zip(sets, together):
                alone = gc.project_and_classify(K, T, depth, points, 0.1, window)
                for a, b in zip(batched, alone):
                    assert np.array_equal(a, b, equal_nan=True)
    assert gc.project_and_classify_many(K, np.eye(4), depth, [], 0.1) == []
