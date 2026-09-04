"""The node's asynchronous perception path: frames handed to the detector
thread are fused once their detections return, other frames fuse at once,
map outputs are published at the publish rate, and the published point
cloud carries the map's points."""
import time

import numpy as np
from std_msgs.msg import Header

from semantic_mapping.ros_msgs import pointcloud_to_xyz
from test.ros.helpers import ReplayDetector, feed_frame, stamp_msg


def test_deferred_detection_frames_fuse_and_outputs_publish(node_factory, dataset):
    frames = list(dataset)[:4]
    node = node_factory("-p", "detector_rate_hz:=5.0", "-p", "publish_rate_hz:=5.0", silence_publishers=False)
    node.detector = ReplayDetector(node.detector)

    published = {"boxes": [], "points": [], "annotated": []}
    node.obj_boxes_pub.publish = published["boxes"].append
    node.obj_points_pub.publish = published["points"].append
    node.annotated_image_pub.publish = published["annotated"].append
    node.annotated_image_pub.get_subscription_count = lambda: 1

    for frame in frames:
        feed_frame(node, dataset, frame)
        deadline = time.time() + 5.0
        while node._detection_results.empty() and not node._detection_jobs.empty() and time.time() < deadline:
            time.sleep(0.05)
    time.sleep(0.5)
    node._drain_detection_results(Header(stamp=stamp_msg(frames[-1].stamp), frame_id="map"))

    assert node.detector.calls, "detector thread never ran"
    assert node.detector.calls[0] == 0, "the first frame goes to the detector"
    assert published["annotated"] and published["boxes"] and published["points"]
    objects = node.pipeline.object_map.objects
    assert {"table", "sofa", "shelf"} <= {o.label for o in objects.values()}
    assert all(o.hits >= 1 for o in objects.values())

    # The published cloud is the map's points, label-coloured, in the world frame.
    result = node._last_result
    node._publish_result(result, Header(stamp=stamp_msg(frames[-1].stamp), frame_id="map"))
    cloud = published["points"][-1]
    expected = np.concatenate([o.points_world for o in result.objects if o.points_world.shape[0]])
    assert cloud.header.frame_id == node.world_frame and cloud.width == expected.shape[0]
    assert [f.name for f in cloud.fields] == ["x", "y", "z", "rgb"]
    np.testing.assert_allclose(pointcloud_to_xyz(cloud), expected, atol=1e-5)

    logged = []
    node.get_logger().info = lambda message, **kwargs: logged.append(message)
    node._log_runtime_stats()
    assert logged and logged[-1].startswith("runtime:")
