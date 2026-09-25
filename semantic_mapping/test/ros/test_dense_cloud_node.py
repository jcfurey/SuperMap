"""Real ROS messages through the independent full-cloud node; no live sensors."""
import json
import threading

import numpy as np
import pytest
import rclpy
from sensor_msgs_py import point_cloud2
from diagnostic_msgs.msg import DiagnosticStatus
from std_msgs.msg import Header, String

from semantic_mapping.dense_cloud_io import full_pointcloud_xyz
from semantic_mapping.dense_cloud_node import DenseCloudMappingNode
from test.ros.helpers import stamp_msg


BASE_ARGS = ["--ros-args", "-p", "cloud_topic:=test/dense_cloud", "-p", "input_mode:=scan",
             "-p", "world_frame:=map", "-p", "min_region_voxels:=1", "-p", "camera_annotation_queue_size:=8",
             "-p", "publish_period_s:=0.0", "-p", "tf_max_wait_s:=0.0"]


def capture_publishers(node):
    node.published_clouds, node.published_maps, node.published_metadata = [], [], []
    node.cloud_pub.publish = node.published_clouds.append
    node.map_pub.publish = node.published_maps.append
    node.regions_pub.publish = node.published_metadata.append


def count(node, name):
    return node._stats.get(name)


@pytest.fixture
def make_dense_node():
    nodes = []

    def make(*extra):
        rclpy.init(args=[*BASE_ARGS, *extra])
        node = DenseCloudMappingNode()
        node.autostart_now()
        capture_publishers(node)
        nodes.append(node)
        return node

    yield make
    for node in nodes:
        node.destroy_node()
    rclpy.try_shutdown()


@pytest.fixture
def dense_node(make_dense_node):
    return make_dense_node()


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
    regions = node.published_metadata[-1]
    assert regions.header.frame_id == "map" and regions.labels[:2] == ["unknown", "chair"]
    assert sum(r.voxel_count for r in regions.regions) == 4


def test_stale_camera_and_unapproved_source_keep_full_geometry(dense_node):
    node = dense_node
    node._on_cloud(cloud([[0., 0., 2.], [10., 0., 2.]]))
    before = node.pipeline.result()
    node._on_annotations(manual_labels(stamp=.1))
    node._on_annotations(manual_labels(source="unapproved_model"))
    assert count(node, "annotations_rejected") == 2
    np.testing.assert_array_equal(node.pipeline.result().points_world, before.points_world)
    np.testing.assert_array_equal(node.pipeline.result().semantic_ids, before.semantic_ids)


def test_missing_tf_never_falls_back_to_latest_or_identity(dense_node):
    node = dense_node
    node._on_cloud(cloud([[0., 0., 2.]]))
    node._on_cloud(cloud([[100., 0., 2.]], stamp=2., frame="uncalibrated_sensor"))
    assert count(node, "clouds_rejected") == 1
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
    assert count(node, "clouds_rejected") == 1 and node.pipeline.stamp == 1.


def test_zero_stamp_never_requests_latest_dynamic_tf(dense_node):
    node = dense_node
    node._on_cloud(cloud([[0., 0., 2.]], stamp=0., frame="sensor"))
    assert count(node, "clouds_rejected") == 1 and not node.published_clouds


def test_annotations_arriving_before_their_cloud_are_queued_and_applied(dense_node):
    node = dense_node
    node._on_annotations(manual_labels(stamp=2.))
    assert len(node._pending_annotations) == 1 and not node.published_clouds
    node._on_cloud(cloud([[0., 0., 2.], [10., 0., 2.]], stamp=1.))
    assert node.pipeline.result().semantic_ids.tolist() == [0, 0]
    node._on_cloud(cloud([[0., 0., 2.], [10., 0., 2.]], stamp=2.))
    assert node.pipeline.result().semantic_ids.tolist() == [1, 0]
    assert not node._pending_annotations and count(node, "annotations_applied") == 1


def test_queued_annotations_are_bounded_and_expire_without_labeling_newer_cloud(dense_node):
    node = dense_node
    for stamp in range(1, 21):
        node._on_annotations(manual_labels(stamp=float(stamp)))
    assert len(node._pending_annotations) == 8
    node._on_cloud(cloud([[0., 0., 2.]], stamp=30.))
    assert node.pipeline.result().semantic_ids.tolist() == [0]
    assert not node._pending_annotations and count(node, "annotations_expired") == 20


def test_mask_reception_continues_while_cloud_segmentation_is_busy(dense_node, monkeypatch):
    node = dense_node
    started, release = threading.Event(), threading.Event()
    original = node.pipeline.update

    def slow_update(*args):
        started.set()
        assert release.wait(5.0)
        return original(*args)

    monkeypatch.setattr(node.pipeline, "update", slow_update)
    worker = threading.Thread(target=node._on_cloud, args=(cloud([[0., 0., 2.], [10., 0., 2.]]),))
    worker.start()
    try:
        assert started.wait(5.0)
        node._on_annotations(manual_labels())
        assert len(node._pending_annotations) == 1 and count(node, "annotations_applied") == 0
    finally:
        release.set()
        worker.join(5.0)
    assert not worker.is_alive()
    assert node.pipeline.result().semantic_ids.tolist() == [1, 0]
    assert count(node, "annotations_applied") == 1


@pytest.mark.parametrize("bad", [
    {"detections": ["chair"]}, {"detections": [{"label": ["chair"], "mask_indices": [55]}]},
    {"detections": [{"label": "chair", "mask_runs": "55"}]}, {"detections": {"label": "chair"}},
    {"intrinsics": "fx"}, {"stamp": "1.0"}, {"frame_id": 7},
])
def test_malformed_manual_annotations_are_rejected_without_crashing(dense_node, bad):
    node = dense_node
    node._on_cloud(cloud([[0., 0., 2.]]))
    payload = json.loads(manual_labels().data)
    payload.update(bad)
    node._on_annotations(String(data=json.dumps(payload)))
    node._on_annotations(String(data="[1, 2"))
    node._on_annotations(String(data="[" * 100000 + "]" * 100000))
    assert count(node, "annotations_rejected") == 3
    node._on_annotations(manual_labels())
    assert node.pipeline.result().semantic_ids.tolist() == [1]


def test_annotation_size_and_count_limits(make_dense_node):
    node = make_dense_node("-p", "max_annotation_bytes:=2000", "-p", "max_detections:=2")
    node._on_cloud(cloud([[0., 0., 2.]]))
    payload = json.loads(manual_labels().data)
    payload["detections"] = payload["detections"] * 3
    node._on_annotations(String(data=json.dumps(payload)))
    payload["detections"] = [{"label": "chair", "mask_indices": list(range(100))}] * 2
    payload["padding"] = "x" * 3000
    node._on_annotations(String(data=json.dumps(payload)))
    assert count(node, "annotations_rejected") == 2 and count(node, "annotations_applied") == 0


def test_label_vocabulary_cap_is_a_parameter(make_dense_node):
    node = make_dense_node("-p", "max_labels:=2")
    node._on_cloud(cloud([[0., 0., 2.]]))
    node._on_annotations(manual_labels())
    payload = json.loads(manual_labels().data)
    payload["detections"][0]["label"] = "table"
    node._on_annotations(String(data=json.dumps(payload)))
    assert count(node, "annotations_applied") == 1 and count(node, "annotations_rejected") == 1


def test_cloud_waits_for_late_tf_instead_of_being_dropped(make_dense_node):
    from geometry_msgs.msg import TransformStamped

    node = make_dense_node("-p", "tf_max_wait_s:=30.0", "-p", "tf_timeout_s:=0.0")
    node._on_cloud(cloud([[0., 0., 2.]], stamp=5., frame="late_sensor"))
    assert len(node._tf_waiting) == 1 and not node.published_clouds and count(node, "clouds_rejected") == 0
    tf = TransformStamped()
    tf.header = Header(stamp=stamp_msg(5.), frame_id="map")
    tf.child_frame_id = "late_sensor"
    tf.transform.translation.x = 10.
    tf.transform.rotation.w = 1.
    node.tf_buffer.set_transform_static(tf, "test")
    node._retry_tf_queue()
    assert not node._tf_waiting and len(node.published_clouds) == 1
    np.testing.assert_allclose(node.pipeline.result().map_points_world, [[10., 0., 2.]], atol=0.05)


def test_tf_queue_is_bounded_and_expired_clouds_are_rejected(make_dense_node):
    node = make_dense_node("-p", "tf_max_wait_s:=30.0", "-p", "tf_timeout_s:=0.0", "-p", "tf_queue_size:=2")
    for stamp in (1., 2., 3.):
        node._on_cloud(cloud([[0., 0., 2.]], stamp=stamp, frame="never"))
    assert len(node._tf_waiting) == 2 and count(node, "clouds_dropped_tf_queue") == 1
    node._tf_max_wait = 0.0
    node._retry_tf_queue()
    assert not node._tf_waiting and count(node, "tf_failures") == 2


def test_map_and_region_publishing_is_throttled(make_dense_node):
    node = make_dense_node("-p", "publish_period_s:=100.0")
    node._on_cloud(cloud([[0., 0., 2.]], stamp=1.))
    node._on_cloud(cloud([[1., 0., 2.]], stamp=2.))
    assert len(node.published_clouds) == 2 and not node.published_maps and not node.published_metadata
    node._on_publish_timer()
    node._on_cloud(cloud([[2., 0., 2.]], stamp=3.))
    node._on_publish_timer()
    assert len(node.published_maps) == 1 and len(node.published_metadata) == 1
    assert len(node.published_metadata[0].regions) == 2  # built from the latest map at publish time


def test_lifecycle_inactive_ignores_input_and_cleanup_reconfigures(dense_node):
    node = dense_node
    node.trigger_deactivate()
    node._on_cloud(cloud([[0., 0., 2.]]))
    assert node.pipeline.stamp == float("-inf") and count(node, "clouds_dropped_inactive") == 1
    node.trigger_cleanup()
    assert node.pipeline is None and node.cloud_sub is None and node.tf_listener is None
    node.trigger_configure()
    node.trigger_activate()
    capture_publishers(node)
    node._on_cloud(cloud([[0., 0., 2.]]))
    assert len(node.published_clouds) == 1


def test_runtime_parameters_apply_and_structural_ones_are_read_only(dense_node):
    from rclpy.parameter import Parameter

    node = dense_node
    assert node.set_parameters([Parameter("min_camera_score", value=0.9)])[0].successful
    assert node.pipeline.config.min_camera_score == 0.9
    assert node.set_parameters([Parameter("min_camera_score", value=1)])[0].successful  # int accepted
    assert not node.set_parameters([Parameter("min_camera_score", value=2.0)])[0].successful
    assert not node.set_parameters([Parameter("voxel_size", value=0.2)])[0].successful
    assert node.pipeline.config.voxel_size == 0.05


def test_diagnostics_report_counters(dense_node):
    from diagnostic_updater import DiagnosticStatusWrapper

    node = dense_node
    node._on_cloud(cloud([[0., 0., 2.]]))
    node._on_cloud(cloud([[0., 0., 2.]], stamp=2., frame="missing"))
    status = node._diagnose(DiagnosticStatusWrapper())
    values = {item.key: item.value for item in status.values}
    assert values["clouds_processed"] == "1" and values["tf_failures"] == "1"
    assert status.level == DiagnosticStatus.WARN
