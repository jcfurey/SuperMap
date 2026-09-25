"""Real ROS messages through the independent full-cloud node; no live sensors."""
import json

import numpy as np
import pytest
import rclpy
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String

from semantic_mapping.dense_cloud_io import full_pointcloud_xyz
from semantic_mapping.dense_cloud_node import DenseCloudMappingNode
from test.ros.helpers import stamp_msg


@pytest.fixture
def dense_node():
    rclpy.init(args=["--ros-args", "-p", "cloud_topic:=/test/dense_cloud", "-p", "input_mode:=scan",
                    "-p", "world_frame:=map", "-p", "min_region_voxels:=1"])
    node = DenseCloudMappingNode()
    node.published_clouds, node.published_maps, node.published_metadata = [], [], []
    node.cloud_pub.publish = node.published_clouds.append
    node.map_pub.publish = node.published_maps.append
    node.regions_pub.publish = node.published_metadata.append
    yield node
    node.destroy_node()
    rclpy.shutdown()


def cloud(points, stamp=1., frame="map"):
    return point_cloud2.create_cloud_xyz32(Header(stamp=stamp_msg(stamp), frame_id=frame), np.asarray(points, np.float32))


def manual_labels(**overrides):
    payload = {"source": "manual", "rectified": True, "stamp": 1., "frame_id": "map",
               "intrinsics": {"fx": 10., "fy": 10., "cx": 5., "cy": 5., "width": 10, "height": 10},
               "detections": [{"label": "chair", "mask_indices": [55]}]}
    payload.update(overrides)
    return String(data=json.dumps(payload))


def test_cloud_publishes_without_camera_then_receives_partial_camera_labels(dense_node):
    node = dense_node
    points = [[0., 0., 2.], [0., 0., 4.], [10., 0., 2.], [np.nan, 0., 0.]]
    node._on_cloud(cloud(points))
    assert len(node.published_clouds) == 1
    assert node.published_clouds[-1].width == 4
    np.testing.assert_equal(full_pointcloud_xyz(node.published_clouds[-1]), points)
    assert node.pipeline.result().semantic_ids.tolist() == [0, 0, 0, 0]
    node._on_annotations(manual_labels())
    assert node.pipeline.result().semantic_ids.tolist() == [1, 0, 0, 0]
    assert len(node.published_clouds) == 2
    node._on_cloud(cloud([[20., 0., 2.]], stamp=2.))
    assert len(node.pipeline.result().map_points_world) == 4
    assert (node.pipeline.result().map_semantic_ids > 0).sum() == 1
    metadata = json.loads(node.published_metadata[-1].data)
    assert metadata["world_frame"] == "map" and metadata["stats"]["pretrained_models"] == []


def test_stale_camera_and_unapproved_source_keep_full_geometry(dense_node):
    node = dense_node
    node._on_cloud(cloud([[0., 0., 2.], [10., 0., 2.]]))
    before = node.pipeline.result()
    node._on_annotations(manual_labels(stamp=.1))
    node._on_annotations(manual_labels(source="unapproved_model"))
    assert node._rejected_annotations == 2
    np.testing.assert_array_equal(node.pipeline.result().points_world, before.points_world)
    np.testing.assert_array_equal(node.pipeline.result().semantic_ids, before.semantic_ids)


def test_missing_tf_never_falls_back_to_latest_or_identity(dense_node):
    node = dense_node
    node._on_cloud(cloud([[0., 0., 2.]]))
    node._on_cloud(cloud([[100., 0., 2.]], stamp=2., frame="uncalibrated_sensor"))
    assert node._rejected_clouds == 1
    assert node.pipeline.stamp == 1.
    np.testing.assert_allclose(node.pipeline.result().points_world, [[0., 0., 2.]])


def test_empty_annotations_are_not_free_space_or_negative_object_evidence(dense_node):
    node = dense_node
    node._on_cloud(cloud([[0., 0., 2.], [10., 0., 2.]]))
    node._on_annotations(manual_labels())
    node._on_annotations(manual_labels(detections=[]))
    assert node.pipeline.result().semantic_ids.tolist() == [1, 0]
    assert len(node.pipeline.result().points_world) == 2


def test_feedback_cloud_rejected_before_map_mutation(dense_node):
    node = dense_node
    node._on_cloud(cloud([[0., 0., 2.]]))
    feedback = node.published_clouds[-1]
    feedback.header.stamp = stamp_msg(2.)
    node._on_cloud(feedback)
    assert node._rejected_clouds == 1 and node.pipeline.stamp == 1.


def test_zero_stamp_never_requests_latest_dynamic_tf(dense_node):
    node = dense_node
    node._on_cloud(cloud([[0., 0., 2.]], stamp=0., frame="sensor"))
    assert node._rejected_clouds == 1 and not node.published_clouds
