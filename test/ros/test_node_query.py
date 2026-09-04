"""Language grounding through the node: an instruction on the query topic
yields the answer JSON, a nav_msgs/Path of waypoints, and a PoseStamped goal."""
import json
import time

import numpy as np
from std_msgs.msg import String

from test.ros.helpers import feed_frames


def _wait(published, count, timeout=5.0):
    deadline = time.time() + timeout
    while len(published) < count and time.time() < deadline:
        time.sleep(0.05)


def test_query_publishes_answer_path_and_goal(node_factory, dataset):
    node = node_factory("-p", "vlm.client:=keyword")
    published = {"answer": [], "goal": [], "waypoints": []}
    node.answer_pub.publish = published["answer"].append
    node.goal_pub.publish = published["goal"].append
    node.waypoints_pub.publish = published["waypoints"].append

    feed_frames(node, dataset, list(dataset)[:6])
    labels = {o.label for o in node.pipeline.object_map.objects.values()}
    assert {"sofa", "shelf"} <= labels

    node._on_query(String(data="go to the sofa, then the shelf"))
    _wait(published["answer"], 1)
    assert published["answer"], "no answer published"
    answer = json.loads(published["answer"][0].data)
    assert answer["error"] is None and len(answer["target_ids"]) == 2
    assert published["goal"] and published["waypoints"]
    path = published["waypoints"][0]
    assert path.header.frame_id == "map" and len(path.poses) == 2
    goal = published["goal"][0].pose.position
    assert np.allclose([goal.x, goal.y, goal.z], answer["waypoints"][0])

    node._on_query(String(data="find the fridge"))
    _wait(published["answer"], 2)
    failed = json.loads(published["answer"][1].data)
    assert failed["error"] and len(published["goal"]) == 1  # no goal for an unresolved query
