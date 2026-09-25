"""Language grounding through the node.

The GroundInstruction action is the primary interface; the legacy String
query topic still answers with JSON (now carrying a request ID). Goals are
approach poses beside the object, never its occupied centroid.
"""
import json
import math
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from std_msgs.msg import String
from supermap_msgs.action import GroundInstruction

from semantic_mapping.node import approach_pose
from test.ros.helpers import BackgroundExecutor, feed_frames, wait_for


def _wait(published, count, timeout=5.0):
    deadline = time.time() + timeout
    while len(published) < count and time.time() < deadline:
        time.sleep(0.05)


def _footprint_distance(point, bbox):
    dx = max(bbox[0] - point[0], 0.0, point[0] - bbox[3])
    dy = max(bbox[1] - point[1], 0.0, point[1] - bbox[4])
    return math.hypot(dx, dy)


def _faces(pose_xy_yaw, bbox):
    x, y, yaw = pose_xy_yaw
    cx, cy = (bbox[0] + bbox[3]) / 2, (bbox[1] + bbox[4]) / 2
    return abs(math.remainder(yaw - math.atan2(cy - y, cx - x), 2 * math.pi)) < 1e-6


# ------------------------------------------------------------ approach poses
def test_approach_pose_stands_off_towards_the_robot_and_faces_the_object():
    bbox = [0, 0, 0, 1, 1, 1]
    x, y, yaw = approach_pose(bbox, robot_xy=np.array([5.0, 0.5]), standoff=0.6)
    assert (x, y) == pytest.approx((1.6, 0.5)) and yaw == pytest.approx(math.pi)
    assert _footprint_distance((x, y), bbox) == pytest.approx(0.6)


def test_approach_pose_avoids_other_objects_and_uses_the_supporting_furniture():
    bbox = [0, 0, 0, 1, 1, 1]
    blocker = [1.3, -0.5, 0, 2.5, 1.5, 1]  # occupies the side facing the robot
    pose = approach_pose(bbox, robot_xy=np.array([5.0, 0.5]), standoff=0.6, obstacles=[blocker], clearance=0.2)
    assert not (1.1 <= pose[0] <= 2.7 and -0.7 <= pose[1] <= 1.7)
    assert _faces(pose, bbox) and _footprint_distance(pose[:2], bbox) >= 0.6 - 1e-9

    mug, table = [0.4, 0.4, 0.8, 0.6, 0.6, 0.9], [0, 0, 0, 2, 1, 0.8]
    pose = approach_pose(mug, robot_xy=np.array([5.0, 0.5]), standoff=0.6, obstacles=[table])
    assert pose[:2] == pytest.approx((2.6, 0.5)) and _faces(pose, mug)

    floor = [-50, -50, -0.1, 50, 50, 0]  # too large to count as an obstacle or a support
    assert approach_pose(bbox, np.array([5.0, 0.5]), 0.6, [floor])[:2] == pytest.approx((1.6, 0.5))


def test_approach_pose_without_robot_uses_the_narrow_side():
    bbox = [0, 0, 0, 0.2, 2, 1]
    x, y, yaw = approach_pose(bbox, robot_xy=None, standoff=0.5)
    assert (x, y) == pytest.approx((0.7, 1.0)) and _faces((x, y, yaw), bbox)


# ----------------------------------------------------------------- legacy topic
def test_query_publishes_answer_path_and_goal(node_factory, dataset):
    node = node_factory("-p", "vlm.client:=keyword", "-p", "goal_standoff_m:=0.5")
    published = {"answer": [], "goal": [], "waypoints": []}
    node.answer_pub.publish = published["answer"].append
    node.goal_pub.publish = published["goal"].append
    node.waypoints_pub.publish = published["waypoints"].append

    feed_frames(node, dataset, list(dataset)[:6])
    objects = {o.instance_id: o for o in node.pipeline.object_map.objects.values()}
    assert {"sofa", "shelf"} <= {o.label for o in objects.values()}

    node._on_query(String(data=json.dumps({"instruction": "go to the sofa, then the shelf", "request_id": "r1"})))
    _wait(published["answer"], 1)
    assert published["answer"], "no answer published"
    answer = json.loads(published["answer"][0].data)
    assert answer["request_id"] == "r1"
    assert answer["error"] is None and len(answer["target_ids"]) == 2 and len(answer["goals"]) == 2
    assert published["goal"] and published["waypoints"]
    path = published["waypoints"][0]
    assert path.header.frame_id == "map" and len(path.poses) == 2
    for target, pose in zip(answer["target_ids"], path.poses):
        bbox = objects[target].bbox3d
        position, q = pose.pose.position, pose.pose.orientation
        # Not the occupied centroid: outside the footprint by the standoff, facing the object.
        assert _footprint_distance((position.x, position.y), bbox) >= 0.5 - 1e-6
        assert _faces((position.x, position.y, 2 * math.atan2(q.z, q.w)), bbox)
    goal = published["goal"][0].pose.position
    assert np.allclose([goal.x, goal.y, goal.z], answer["goals"][0][:3])
    assert not np.allclose([goal.x, goal.y, goal.z], answer["waypoints"][0])

    node._on_query(String(data="find the fridge"))
    _wait(published["answer"], 2)
    failed = json.loads(published["answer"][1].data)
    assert failed["error"] and failed["request_id"].startswith("query-")
    assert len(published["goal"]) == 1  # no goal for an unresolved query


def test_robot_pose_uses_the_latest_transform(node_factory, dataset):
    """C7: the lookup is not made at now(), which is always ahead of the newest transform."""
    node = node_factory()
    assert node._robot_position() is None  # no TF yet: warn and fall back, never raise
    frames = list(dataset)[:2]
    feed_frames(node, dataset, frames)
    assert node.get_clock().now().nanoseconds * 1e-9 > frames[-1].stamp + 1.0
    np.testing.assert_allclose(node._robot_position(), frames[-1].T_world_from_cam[:3, 3])


# ----------------------------------------------------------------------- action
@pytest.fixture
def action_setup(node_factory, dataset):
    def make(*ros_args):
        node = node_factory("-p", "vlm.client:=keyword", *ros_args)
        for publisher in (node.answer_pub, node.goal_pub, node.waypoints_pub):
            publisher.publish = lambda message: None
        feed_frames(node, dataset, list(dataset)[:6])
        client_node = rclpy.create_node("grounding_test_client")
        client = ActionClient(client_node, GroundInstruction, "/semantic_mapping_node/ground_instruction")
        return node, client_node, client

    return make


def _send(client, instruction, feedback=None, radius=0.0):
    future = client.send_goal_async(GroundInstruction.Goal(instruction=instruction, local_radius_m=radius),
                                    feedback_callback=feedback)
    wait_for(future.done)
    return future.result()


def _result(goal_handle, timeout=10.0):
    future = goal_handle.get_result_async()
    wait_for(future.done, timeout)
    return future.result()


def test_action_grounds_instruction_with_feedback_and_approach_poses(action_setup):
    node, client_node, client = action_setup()
    stages = []
    try:
        with BackgroundExecutor(node, client_node):
            assert client.wait_for_server(timeout_sec=5.0)
            handle = _send(client, "go to the sofa, then the shelf", lambda msg: stages.append(msg.feedback.stage))
            assert handle.accepted
            response = _result(handle)
            assert response.status == GoalStatus.STATUS_SUCCEEDED
            result = response.result
            assert result.success and list(result.target_labels) == ["sofa", "shelf"]
            assert len(result.goals) == 2 and len(result.path.poses) == 2
            assert result.goals[0].header.frame_id == "map"
            objects = node.pipeline.object_map.objects
            for target, goal in zip(result.target_ids, result.goals):
                p = goal.pose.position
                assert _footprint_distance((p.x, p.y), objects[target].bbox3d) >= 0.6 - 1e-6
            wait_for(lambda: "parsing" in stages)
            assert stages[:2] == ["serializing", "queued"] and "querying" in stages

            # A tiny local radius around the robot (C7 pose) leaves nothing to ground.
            response = _result(_send(client, "go to the sofa", radius=0.01))
            assert response.status == GoalStatus.STATUS_ABORTED and not response.result.success

            handle = _send(client, "   ")
            assert not handle.accepted
    finally:
        client.destroy()
        client_node.destroy_node()


def test_action_cancel_and_busy_rejection(action_setup):
    node, client_node, client = action_setup("-p", "grounding_max_queue:=1")
    release = threading.Event()
    inner = node.grounder.client

    def slow_complete(prompt):
        release.wait(10.0)
        return inner.complete(prompt)

    node.grounder.client = SimpleNamespace(complete=slow_complete)
    try:
        with BackgroundExecutor(node, client_node):
            assert client.wait_for_server(timeout_sec=5.0)
            first = _send(client, "go to the sofa")
            assert first.accepted
            wait_for(lambda: node._grounding_outstanding == 1)
            assert not _send(client, "go to the shelf").accepted  # queue bound reached

            cancel = first.cancel_goal_async()
            wait_for(cancel.done)
            assert _result(first).status == GoalStatus.STATUS_CANCELED
            release.set()
            wait_for(lambda: node._grounding_outstanding == 0)
            response = _result(_send(client, "go to the shelf"))
            assert response.status == GoalStatus.STATUS_SUCCEEDED
    finally:
        release.set()
        client.destroy()
        client_node.destroy_node()


def test_runtime_tunable_grounding_parameters(node_factory):
    from rclpy.parameter import Parameter

    node = node_factory()
    assert node.set_parameters([Parameter("goal_standoff_m", value=1)])[0].successful  # int accepted for a double
    assert node._goal_standoff_m == 1.0
    assert not node.set_parameters([Parameter("goal_standoff_m", value=-1.0)])[0].successful
    assert node.set_parameters([Parameter("vlm.local_radius_m", value=3.0)])[0].successful
    assert node.grounder.local_radius_m == 3.0
    assert not node.set_parameters([Parameter("world_frame", value="odom")])[0].successful  # read-only
