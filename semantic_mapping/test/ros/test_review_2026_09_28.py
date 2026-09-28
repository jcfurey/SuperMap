"""Lifecycle, grounding and map-output regressions from the repository review."""
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from rclpy.action import GoalResponse
from std_msgs.msg import Header, String
from supermap_msgs.action import GroundInstruction

from semantic_mapping.node import _GroundingOutcome
from semantic_mapping.pipeline import FrameResult
from semantic_mapping.ros_node_utils import TransitionCallbackReturn
from semantic_mapping.scene_graph import SceneGraph
from semantic_mapping.types import Detection2D
from test.helpers import make_object
from test.ros.helpers import spin_until, wait_for
from test.ros.test_node_lifecycle_and_outputs import _depth_node, _feed_depth
from test.test_pipeline import _observation


class BlockingDetector:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def detect(self, image, **kwargs):
        self.started.set()
        assert self.release.wait(10.)
        return [Detection2D(np.array([60., 40., 100., 80.]), 'old session', .9)]


def _map_for_grounding(node):
    near = make_object(1, 'chair', [-.5, -.5, 1., .5, .5, 2.])
    far = make_object(2, 'table', [10., 0., 1., 11., 1., 2.])
    node._last_result = FrameResult(objects=[near, far], stamp=10., scene_graph=SceneGraph(node_ids=[1, 2]))
    node._robot_position = lambda: np.zeros(3)
    node.grounder.local_radius_m = 2.
    outputs = {name: [] for name in ('answer', 'goal', 'waypoints', 'nav2')}
    for name in ('answer', 'goal', 'waypoints'):
        getattr(node, name + '_pub').publish = outputs[name].append
    node._nav2_send_goal = True
    node._send_nav2_goal = outputs['nav2'].append
    return outputs


@pytest.mark.parametrize('worker_kind', ['detector', 'grounding'])
def test_cleanup_retains_busy_worker_and_refuses_model_replacement(node_factory, worker_kind):
    node = _depth_node(node_factory)
    blocker = BlockingDetector()
    outputs = _map_for_grounding(node)
    if worker_kind == 'detector':
        node.detector = blocker
        _feed_depth(node, _observation(10., 2., False))
        worker = node._detector_thread
    else:
        def complete(prompt):
            blocker.started.set()
            assert blocker.release.wait(10.)
            return '<answer>1</answer>'

        node.grounder.client = SimpleNamespace(complete=complete)
        node._on_query(String(data='go to chair'))
        worker = node._grounding_thread
    try:
        assert blocker.started.wait(2.)
        stop_event, old_pipeline = node._stop_event, node.pipeline
        assert node.trigger_deactivate() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_cleanup() == TransitionCallbackReturn.SUCCESS
        assert worker.is_alive() and stop_event.is_set()
        assert getattr(node, '_' + worker_kind + '_thread') is worker
        assert node.trigger_configure() == TransitionCallbackReturn.FAILURE
        assert node._stop_event is stop_event and node.pipeline is old_pipeline
    finally:
        blocker.release.set()
        worker.join(3.)
    assert not worker.is_alive()
    assert not any(outputs.values())
    assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
    assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
    assert node._stop_event is not stop_event and node.pipeline is not old_pipeline
    node._lookup_se3 = lambda *args: np.eye(4)
    node.detector = SimpleNamespace(detect=lambda *args, **kwargs: [
        Detection2D(np.array([60., 40., 100., 80.]), 'new session', .9)])
    _feed_depth(node, _observation(30., 2., False))
    spin_until(node, lambda: node._last_result is not None)
    assert [o.label for o in node._last_result.objects] == ['new session']
    assert node._last_result.stamp == 30.


@pytest.mark.parametrize('reactivate_before_completion', [False, True])
def test_deactivation_discards_pending_frames_across_reactivation(node_factory, reactivate_before_completion):
    node = _depth_node(node_factory)
    detector = BlockingDetector()
    node.detector = detector
    _feed_depth(node, _observation(10., 2., False))
    try:
        assert detector.started.wait(2.)
        node.trigger_deactivate()
        if reactivate_before_completion:
            node.trigger_activate()
        detector.release.set()
        wait_for(lambda: not node._detection_results.empty())
        node._drain_detection_results()
        assert node.pipeline._frame_index == 0 and node._last_result is None
        assert not node._pending_frames and not node._tf_waiting
        if not reactivate_before_completion:
            node.trigger_activate()
        node.detector = SimpleNamespace(detect=lambda *args, **kwargs: [])
        _feed_depth(node, _observation(11., 2., False))
        spin_until(node, lambda: node.pipeline._frame_index == 1)
        assert not node._last_result.objects
    finally:
        detector.release.set()


@pytest.mark.parametrize('reactivate_before_completion', [False, True])
def test_deactivation_invalidates_running_and_queued_grounding(node_factory, reactivate_before_completion):
    node = node_factory()
    outputs = _map_for_grounding(node)
    started, release = threading.Event(), threading.Event()
    calls = []

    def complete(prompt):
        calls.append(prompt)
        started.set()
        assert release.wait(10.)
        return '<answer>1</answer>'

    node.grounder.client = SimpleNamespace(complete=complete)
    node._on_query(String(data='first chair request'))
    try:
        assert started.wait(2.)
        node._on_query(String(data='queued chair request'))
        node.trigger_deactivate()
        if reactivate_before_completion:
            node.trigger_activate()
        release.set()
        wait_for(lambda: node._grounding_outstanding == 0)
        assert len(calls) == 1 and not any(outputs.values())
        if not reactivate_before_completion:
            node.trigger_activate()
        node._on_query(String(data='fresh chair request'))
        wait_for(lambda: len(outputs['nav2']) == 1)
        assert len(calls) == 2
    finally:
        release.set()


@pytest.mark.parametrize('answer, expected', [('2', []), ('2, 1, 999', [1]), ('1, 2', [1])])
def test_grounding_only_navigates_to_validated_candidates(node_factory, answer, expected):
    node = node_factory()
    outputs = _map_for_grounding(node)
    node.grounder.client = SimpleNamespace(complete=lambda prompt: f'<answer>{answer}</answer>')
    job = node._prepare_grounding('go to chair', 'review')
    assert list(job.bboxes) == [1] and set(job.obstacles) == {1, 2}
    result = node.grounder.complete(job.request)
    outcome = node._finish_grounding(job, result)
    assert outcome.target_ids == expected
    assert outcome.target_labels == (['chair'] if expected else [])
    assert len(outcome.goals) == len(expected) and outcome.success == bool(expected)
    assert 2 in outcome.unresolved_ids
    node._on_query(String(data='go to chair'))
    wait_for(lambda: node._grounding_outstanding == 0)
    payload = json.loads(outputs['answer'][0].data)
    assert len(payload['goals']) == len(expected)
    assert len(outputs['nav2']) == len(outputs['goal']) == len(outputs['waypoints']) == bool(expected)


def test_failed_grounding_outcome_never_publishes_even_with_coordinates(node_factory):
    node = node_factory()
    outputs = _map_for_grounding(node)
    node._publish_goals(_GroundingOutcome(False, 'failed', goals=[(1., 2., 0., 0.)]))
    assert not any(outputs.values())


@pytest.mark.parametrize('reconfigure', [False, True])
def test_accepted_action_cannot_start_in_a_later_lifecycle_epoch(node_factory, reconfigure):
    node = node_factory()
    outputs = _map_for_grounding(node)
    request = GroundInstruction.Goal(instruction='go to chair')
    assert node._on_ground_goal(request) == GoalResponse.ACCEPT
    node.trigger_deactivate()
    if reconfigure:
        node.trigger_cleanup()
        node.trigger_configure()
    node.trigger_activate()
    outputs = _map_for_grounding(node)
    calls, aborted = [], []
    node.grounder.client = SimpleNamespace(complete=lambda prompt: calls.append(prompt))
    # A fresh reservation must not be released by an old action callback.
    fresh = GroundInstruction.Goal(instruction='fresh chair request')
    assert node._on_ground_goal(fresh) == GoalResponse.ACCEPT
    handle = SimpleNamespace(request=request, abort=lambda: aborted.append(True))
    result = node._execute_ground_instruction(handle)
    assert aborted and not result.success and not result.goals
    assert node._grounding_outstanding == 1 and not calls and not any(outputs.values())


def test_point_cloud_republishes_for_equal_sum_movement(node_factory):
    from sensor_msgs_py import point_cloud2

    node = node_factory()
    obj = make_object(1, 'chair', [0., 0., 1., 2., 2., 3.])
    obj.points_world = np.array([[1., 1., 2.], [2., 2., 3.]])
    frame, header = FrameResult(objects=[obj]), Header(frame_id='map')
    published = []
    node.obj_points_pub.publish = published.append
    node._publish_object_points(frame, header)
    node._publish_object_points(frame, header)
    assert len(published) == 1
    obj.points_world += np.array([.5, -.5, 0.])
    node._publish_object_points(frame, header)
    assert len(published) == 2
    xyz = point_cloud2.read_points_numpy(published[-1], field_names=['x', 'y', 'z'])
    np.testing.assert_array_equal(xyz, obj.points_world)
    obj.points_world = obj.points_world.copy()  # equal content in a new allocation is unchanged
    node._publish_object_points(frame, header)
    assert len(published) == 2
