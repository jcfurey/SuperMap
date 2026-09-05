"""Regressions for chronological fusion, transforms and visualization lifecycle."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker

from semantic_mapping.geometry_utils import invert_se3, rasterize_depth, transform_points
from semantic_mapping.pipeline import FrameResult, PipelineConfig, SemanticMappingPipeline
from semantic_mapping.ros_msgs import numpy_to_image, stamp_to_seconds
from semantic_mapping.scene_graph import SceneGraph
from semantic_mapping.types import ObjectStatus
from test.helpers import make_object
from test.test_pipeline import _observation
from test.ros.helpers import camera_info, set_camera_tf, spin_until, stamp_msg


def _feed_depth(node, observation):
    header = Header(stamp=stamp_msg(observation.stamp), frame_id=node.camera_frame)
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    rgb_msg = numpy_to_image(rgb, 'rgb8', header)
    depth_msg = numpy_to_image(observation.depth.astype(np.float32), '32FC1', header)
    node._on_synced_frame(rgb_msg, camera_info(observation.intrinsics, header), depth_msg, Odometry())
    return header


def _block_detector(node, detections):
    started, release = threading.Event(), threading.Event()

    def detect(*args, **kwargs):
        started.set()
        if not release.wait(5.0):
            raise RuntimeError('test did not release the detector')
        return detections

    node.detector = SimpleNamespace(detect=detect)
    return started, release


def test_late_detection_fuses_before_newer_removal_evidence(node_factory):
    node = node_factory('-p', 'depth_source:=depth_image', '-p', 'detector_timeout_sec:=10.0',
                        '-p', 'min_hits_to_confirm:=1', '-p', 'publish_rate_hz:=100.0')
    node._lookup_se3 = lambda *args: np.eye(4)
    for i in range(5):
        node.pipeline.process_frame(_observation(i * 0.1, 2.0, True))
    deferred = _observation(0.5, 2.0, True)
    started, release = _block_detector(node, deferred.detections)
    processed, annotated, clouds = [], [], []
    original = node._process_and_publish

    def record(obs, header):
        processed.append(obs.stamp)
        return original(obs, header)

    node._process_and_publish = record
    node.annotated_image_pub.get_subscription_count = lambda: 1
    node.annotated_image_pub.publish = annotated.append
    node.obj_points_pub.publish = clouds.append
    try:
        source_header = _feed_depth(node, deferred)
        spin_until(node, started.is_set)
        for i in range(6, 12):
            _feed_depth(node, _observation(i * 0.1, 8.0, False))
        assert node.pipeline._frame_index == 5  # no later evidence can overtake the pending detection
        release.set()
        spin_until(node, lambda: node.pipeline._frame_index == 12)
    finally:
        release.set()

    assert processed == sorted(processed) and len(processed) == 7
    assert node._last_result.stamp == pytest.approx(1.1)
    obj = node.pipeline.object_map.objects[1]
    assert obj.status == ObjectStatus.DISAPPEARED
    assert [s for s, _, _ in obj.trajectory] == sorted(s for s, _, _ in obj.trajectory)
    reference = SemanticMappingPipeline(PipelineConfig(min_hits_to_confirm=1))
    for i in range(12):
        reference.process_frame(_observation(i * 0.1, 2.0 if i <= 5 else 8.0, i <= 5))
    expected = reference.object_map.objects[1]
    assert obj.hits == expected.hits and obj.points_contradicted == expected.points_contradicted
    np.testing.assert_array_equal(obj.point_log_odds, expected.point_log_odds)
    assert annotated and stamp_to_seconds(annotated[-1].header.stamp) == 0.5
    assert annotated[-1].header.frame_id == source_header.frame_id == node.camera_frame
    assert clouds and all(cloud.header.frame_id == node.world_frame for cloud in clouds)


def test_last_detection_is_delivered_without_more_frames_or_clock_updates(node_factory):
    node = node_factory('-p', 'depth_source:=depth_image', '-p', 'use_sim_time:=true')
    node._lookup_se3 = lambda *args: np.eye(4)
    node.detector = SimpleNamespace(detect=lambda *args, **kwargs: _observation(10.0, 2.0, True).detections)
    _feed_depth(node, _observation(10.0, 2.0, False))
    spin_until(node, lambda: node.pipeline._frame_index == 1)
    assert node.get_clock().now().nanoseconds == 0
    assert len(node.pipeline.object_map.objects) == 1


@pytest.mark.parametrize('stamp', [9.9, 10.0])
def test_duplicate_or_older_sensor_frames_are_skipped(node_factory, stamp):
    node = node_factory('-p', 'depth_source:=depth_image')
    node._lookup_se3 = lambda *args: np.eye(4)
    node.detector = SimpleNamespace(detect=lambda *args, **kwargs: [])
    _feed_depth(node, _observation(10.0, 2.0, False))
    spin_until(node, lambda: node.pipeline._frame_index == 1)
    _feed_depth(node, _observation(stamp, 8.0, False))
    assert node.pipeline._frame_index == 1 and node._next_frame_id == 1
    assert node._last_result.stamp == 10.0 and not node._pending_frames


@pytest.mark.parametrize('limit', ['deadline', 'buffer'])
def test_detector_limits_release_geometry_and_discard_late_results(node_factory, limit):
    node = node_factory('-p', 'depth_source:=depth_image', '-p', 'use_sim_time:=true',
                        '-p', f'detector_timeout_sec:={0.08 if limit == "deadline" else 10.0}',
                        '-p', f'max_pending_frames:={2 if limit == "buffer" else 30}')
    node._lookup_se3 = lambda *args: np.eye(4)
    started, release = _block_detector(node, _observation(10.0, 2.0, True).detections)
    try:
        _feed_depth(node, _observation(10.0, 2.0, False))
        spin_until(node, started.is_set)
        expected_frames = 1
        if limit == 'buffer':
            _feed_depth(node, _observation(10.1, 8.0, False))
            _feed_depth(node, _observation(10.2, 8.0, False))
            expected_frames = 3
        spin_until(node, lambda: node.pipeline._frame_index == expected_frames)
        assert not node._pending_frames and not node.pipeline.object_map.objects
        release.set()
        spin_until(node, lambda: node._detector_in_flight is None)
        assert node.pipeline._frame_index == expected_frames
        assert not node.pipeline.object_map.objects
    finally:
        release.set()


def test_invalid_detector_mask_does_not_kill_the_executor(node_factory):
    node = node_factory('-p', 'depth_source:=depth_image')
    node._lookup_se3 = lambda *args: np.eye(4)
    detection = _observation(10.0, 2.0, True).detections[0]
    detection.mask = np.ones((2, 2), dtype=bool)
    node.detector = SimpleNamespace(detect=lambda *args, **kwargs: [detection])
    _feed_depth(node, _observation(10.0, 2.0, False))
    spin_until(node, lambda: node.pipeline._frame_index == 1)
    assert not node.pipeline.object_map.objects


def test_map_reload_invalidates_in_flight_detections(node_factory, tmp_path):
    node = node_factory('-p', 'depth_source:=depth_image')
    node._lookup_se3 = lambda *args: np.eye(4)
    node.pipeline.save(tmp_path / 'empty_map')
    started, release = _block_detector(node, _observation(10.0, 2.0, True).detections)
    try:
        _feed_depth(node, _observation(10.0, 2.0, False))
        spin_until(node, started.is_set)
        node._load_map(str(tmp_path / 'empty_map'))
        release.set()
        spin_until(node, lambda: node._detector_in_flight is None)
        assert node.pipeline._frame_index == 0 and not node.pipeline.object_map.objects
        node.detector = SimpleNamespace(detect=lambda *args, **kwargs: [])
        _feed_depth(node, _observation(10.1, 2.0, False))
        spin_until(node, lambda: node.pipeline._frame_index == 1)
    finally:
        release.set()


@pytest.mark.parametrize('cloud_frame', ['map', 'lidar'])
@pytest.mark.parametrize('accumulate', [1, 3])
def test_cloud_uses_source_time_and_camera_uses_rgb_time(node_factory, cloud_frame, accumulate):
    node = node_factory('-p', f'pointcloud_accumulate_scans:={accumulate}')
    node.detector = SimpleNamespace(detect=lambda *args, **kwargs: [])
    observation = _observation(10.04, 20.0, False)
    T = np.eye(4)
    c, s = np.cos(0.04), np.sin(0.04)
    T[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    for stamp, pose in [(10.0, np.eye(4)), (10.04, T)]:
        set_camera_tf(node, SimpleNamespace(stamp=stamp, T_world_from_cam=pose))
    tf = TransformStamped(header=Header(stamp=stamp_msg(10.0), frame_id='map'), child_frame_id='lidar')
    tf.transform.translation.x = 1.0
    tf.transform.rotation.w = 1.0
    node.tf_buffer.set_transform(tf, 'test')
    points_world = np.array([[0.0, 0.0, 20.0]])
    points = points_world - [1.0, 0.0, 0.0] if cloud_frame == 'lidar' else points_world
    cloud = pc2.create_cloud_xyz32(Header(stamp=stamp_msg(10.0), frame_id=cloud_frame), points.astype(np.float32))
    header = Header(stamp=stamp_msg(10.04), frame_id=node.camera_frame)
    rgb = numpy_to_image(np.zeros((120, 160, 3), dtype=np.uint8), 'rgb8', header)
    captured = []
    original = node._process_and_publish

    def record(obs, header):
        captured.append(obs.depth.copy())
        return original(obs, header)

    node._process_and_publish = record
    node._on_synced_frame(rgb, camera_info(observation.intrinsics, header), cloud, Odometry())
    spin_until(node, lambda: bool(captured))
    expected = rasterize_depth(transform_points(invert_se3(T), points_world), observation.intrinsics.K, 160, 120)
    np.testing.assert_allclose(captured[0], expected)
    assert np.argwhere(captured[0] > 0).tolist() == [[60, 76]]
    if accumulate > 1:
        np.testing.assert_allclose(node._scan_history[0], points_world)


def test_markers_keep_stable_ids_and_delete_removed_instances(node_factory):
    node = node_factory()
    published = []
    node.obj_boxes_pub.publish = published.append
    objects = [make_object(i, 'chair', [i, 0, 0, i + 0.1, 0.1, 0.1]) for i in [1, 2]]
    registry = {}
    for selected in [objects, objects[1:], []]:
        result = FrameResult(objects=selected, stamp=10.0,
                             scene_graph=SceneGraph(node_ids=[o.instance_id for o in selected]))
        node._publish_object_boxes(result, Header(stamp=stamp_msg(10.0), frame_id='map'))
        for marker in published[-1].markers:
            if marker.action == Marker.DELETE:
                registry.pop((marker.ns, marker.id), None)
            else:
                assert marker.action == Marker.ADD
                registry[(marker.ns, marker.id)] = marker
        expected = {(ns, o.instance_id) for o in selected for ns in ['obj_boxes', 'obj_labels']}
        assert set(registry) == expected
    assert registry == {}
