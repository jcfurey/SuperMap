"""Lifecycle, time handling, calibration policy and typed outputs of the live node."""
import math
import os
import subprocess
import sys
import time

import numpy as np
import pytest
import rclpy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
from vision_msgs.msg import Detection3DArray

from semantic_mapping.node import _ScanRing, _label_color
from semantic_mapping.pipeline import FrameResult
from semantic_mapping.ros_msgs import numpy_to_image
from semantic_mapping.ros_node_utils import TransitionCallbackReturn, stable_label_color
from semantic_mapping.scene_graph import SceneGraph
from semantic_mapping.types import ObjectStatus
from test.helpers import make_object
from test.ros.helpers import ROOT, camera_info, set_camera_tf, spin_until, stamp_msg
from test.test_pipeline import _observation


def _feed_depth(node, observation, frame_id=None):
    header = Header(stamp=stamp_msg(observation.stamp), frame_id=frame_id or node.camera_frame)
    rgb_msg = numpy_to_image(np.zeros((120, 160, 3), dtype=np.uint8), 'rgb8', header)
    depth_msg = numpy_to_image(observation.depth.astype(np.float32), '32FC1', header)
    node._on_synced_frame(rgb_msg, camera_info(observation.intrinsics, header), depth_msg)


def _depth_node(node_factory, *args, **kwargs):
    node = node_factory('-p', 'depth_source:=depth_image', *args, **kwargs)
    if getattr(node, '_active', False):
        node._lookup_se3 = lambda *a: np.eye(4)
    return node


# --------------------------------------------------------------- C6 colours
def test_label_colours_are_stable_across_processes():
    code = ("from semantic_mapping.node import _label_color; print(repr(_label_color('chair')))")
    outputs = set()
    for seed in ('1', '2'):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outputs.add(subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True,
                                   check=True, cwd=ROOT).stdout.strip())
    assert len(outputs) == 1
    assert _label_color('chair') == stable_label_color('chair') != stable_label_color('table')


# ------------------------------------------------------------- R8 lifecycle
def test_lifecycle_transitions_gate_input_and_recreate_entities(node_factory):
    node = node_factory('-p', 'depth_source:=depth_image', activate=False)
    assert not hasattr(node, 'pipeline')  # models and I/O load in on_configure, not __init__
    assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
    node._lookup_se3 = lambda *a: np.eye(4)
    _feed_depth(node, _observation(10.0, 2.0, False))
    assert node._next_frame_id == 0  # inactive: input ignored
    assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
    _feed_depth(node, _observation(10.1, 2.0, False))
    spin_until(node, lambda: node._last_result is not None)
    assert node.trigger_deactivate() == TransitionCallbackReturn.SUCCESS
    assert node.trigger_cleanup() == TransitionCallbackReturn.SUCCESS
    assert not node.get_service_names_and_types_by_node(node.get_name(), node.get_namespace()) or all(
        not name.endswith('/save_map') for name, _ in
        node.get_service_names_and_types_by_node(node.get_name(), node.get_namespace()))
    assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS  # the diagnostic updater is reused
    assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
    assert node._active and node._last_result is None


def test_autostart_activates_once_spinning(node_factory):
    node = node_factory(activate=False)
    spin_until(node, lambda: node._active)


def test_configure_fails_without_vocabulary_for_a_real_detector(node_factory, tmp_path):
    empty = tmp_path / 'prompts.yaml'
    empty.write_text('prompts: []\n')
    node = node_factory('-p', 'detector:=yoloe', '-p', f'prompts_file:={empty}', activate=False)
    assert node.trigger_configure() == TransitionCallbackReturn.FAILURE
    node = node_factory('-p', 'detector:=yoloe', '-p', f'prompts_file:={tmp_path / "missing.yaml"}',
                        activate=False)
    assert node.trigger_configure() == TransitionCallbackReturn.FAILURE


def test_relative_prompts_file_resolves_against_the_package_share(node_factory):
    from ament_index_python.packages import get_package_share_directory

    node = node_factory(activate=False)
    resolved = node._resolve_prompts_file('config/prompts.yaml')
    assert resolved.is_file() and str(resolved).startswith(get_package_share_directory('semantic_mapping'))


# ----------------------------------------------------------- C8 time jumps
def test_backwards_clock_jump_resets_input_state(node_factory):
    node = _depth_node(node_factory, '-p', 'use_sim_time:=true')
    clock = node.get_clock()
    clock.set_ros_time_override(Time(seconds=100))
    _feed_depth(node, _observation(100.0, 2.0, False))
    spin_until(node, lambda: node.pipeline._frame_index == 1)
    assert math.isfinite(node._last_input_stamp) and math.isfinite(node._last_detector_stamp)
    clock.set_ros_time_override(Time(seconds=5))  # e.g. ros2 bag play --loop restarting
    assert node._last_input_stamp == -math.inf and node._last_detector_stamp == -math.inf
    assert node._counters['time_jumps'] == 1
    _feed_depth(node, _observation(5.0, 2.0, False))
    spin_until(node, lambda: node.pipeline._frame_index == 2)


def test_backwards_input_stamps_reset_without_sim_time(node_factory):
    node = _depth_node(node_factory)
    _feed_depth(node, _observation(100.0, 2.0, False))
    spin_until(node, lambda: node.pipeline._frame_index == 1)
    _feed_depth(node, _observation(99.9, 2.0, False))  # small reorder: skipped
    assert node._counters['out_of_order'] == 1 and node._next_frame_id == 1
    _feed_depth(node, _observation(5.0, 2.0, False))  # bag restarted: accepted
    spin_until(node, lambda: node.pipeline._frame_index == 2)
    assert node._counters['time_jumps'] == 1


# ------------------------------------------------------- R2 late transforms
def test_frame_waits_for_late_tf_then_drops_after_the_deadline(node_factory, dataset):
    node = node_factory('-p', 'depth_source:=depth_image', '-p', 'tf_wait_sec:=0.3')
    frames = list(dataset)[:2]
    header = Header(stamp=stamp_msg(frames[0].stamp), frame_id=node.camera_frame)
    node._on_synced_frame(numpy_to_image(frames[0].rgb, 'rgb8', header), camera_info(dataset.intrinsics, header),
                          numpy_to_image(frames[0].depth.astype(np.float32), '32FC1', header))
    assert len(node._tf_waiting) == 1 and node._last_result is None
    set_camera_tf(node, frames[0])  # the transform arrives after the images
    spin_until(node, lambda: node._last_result is not None)
    assert node._counters['tf_failures'] == 0

    header = Header(stamp=stamp_msg(frames[1].stamp), frame_id=node.camera_frame)
    node._on_synced_frame(numpy_to_image(frames[1].rgb, 'rgb8', header), camera_info(dataset.intrinsics, header),
                          numpy_to_image(frames[1].depth.astype(np.float32), '32FC1', header))
    spin_until(node, lambda: node._counters['tf_failures'] == 1)
    assert not node._tf_waiting and node.pipeline._frame_index == 1


# --------------------------------------------------------- C29 calibration
def _distorted_info(intr, header):
    info = camera_info(intr, header)
    info.d = [0.1, -0.05, 0.0, 0.0, 0.0]
    info.k = [2 * intr.fx, 0.0, intr.cx, 0.0, 2 * intr.fy, intr.cy, 0.0, 0.0, 1.0]  # raw K differs from P
    info.p = [intr.fx, 0.0, intr.cx, 0.0, 0.0, intr.fy, intr.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return info


def test_distorted_raw_images_are_rejected(node_factory):
    node = _depth_node(node_factory)
    observation = _observation(10.0, 2.0, False)
    header = Header(stamp=stamp_msg(10.0), frame_id=node.camera_frame)
    depth = numpy_to_image(observation.depth.astype(np.float32), '32FC1', header)
    rgb = numpy_to_image(np.zeros((120, 160, 3), np.uint8), 'rgb8', header)
    node._on_synced_frame(rgb, _distorted_info(observation.intrinsics, header), depth)
    assert node._counters['invalid_input'] == 1 and node._next_frame_id == 0


def test_rectified_images_project_with_p(node_factory):
    node = _depth_node(node_factory, '-p', 'camera_images_rectified:=true')
    observation = _observation(10.0, 2.0, False)
    header = Header(stamp=stamp_msg(10.0), frame_id=node.camera_frame)
    info = _distorted_info(observation.intrinsics, header)
    intrinsics, calibration = node._camera_intrinsics(info)
    np.testing.assert_allclose(intrinsics.K, observation.intrinsics.K)
    assert list(calibration.d) == [0.0] * 5 and calibration.k[0] == observation.intrinsics.fx
    rgb = numpy_to_image(np.zeros((120, 160, 3), np.uint8), 'rgb8', header)
    node._on_synced_frame(rgb, info, numpy_to_image(observation.depth.astype(np.float32), '32FC1', header))
    spin_until(node, lambda: node._last_result is not None)
    with pytest.raises(ValueError, match='projection matrix P'):
        node._camera_intrinsics(CameraInfo(width=160, height=120))


# ------------------------------------------------------------ R11 failures
def test_pipeline_exception_is_counted_and_mapping_continues(node_factory):
    node = _depth_node(node_factory)
    original = node.pipeline.process_frame
    calls = []

    def flaky(observation):
        calls.append(observation.stamp)
        if len(calls) == 1:
            raise RuntimeError('boom')
        return original(observation)

    node.pipeline.process_frame = flaky
    _feed_depth(node, _observation(10.0, 2.0, False))
    spin_until(node, lambda: node._counters['failures'] == 1)
    _feed_depth(node, _observation(10.1, 2.0, False))
    spin_until(node, lambda: node._last_result is not None)
    assert node._failures_by_stage == {'mapping': 1}


# -------------------------------------------------- R12 / P6 / R13 outputs
def _result(objects):
    return FrameResult(objects=objects, stamp=10.0, scene_graph=SceneGraph(node_ids=[o.instance_id for o in objects]))


def test_objects_are_published_as_detection3d_array(node_factory):
    node = node_factory()
    published = []
    node.objects_pub.publish = published.append
    chair = make_object(3, 'chair', [0, 0, 0, 1, 2, 3])
    chair.label_belief = {'chair': 0.7, 'sofa': 0.3}
    gone = make_object(4, 'table', [5, 5, 0, 6, 6, 1], status=ObjectStatus.DISAPPEARED)
    node._publish_result(_result([chair, gone]), Header(stamp=stamp_msg(10.0), frame_id='camera'))
    array = published[-1]
    assert isinstance(array, Detection3DArray) and array.header.frame_id == 'map'
    assert [d.id for d in array.detections] == ['3', '4']
    detection = array.detections[0]
    center, size = detection.bbox.center.position, detection.bbox.size
    assert (center.x, center.y, center.z) == (0.5, 1.0, 1.5) and (size.x, size.y, size.z) == (1.0, 2.0, 3.0)
    assert [(r.hypothesis.class_id, r.hypothesis.score) for r in detection.results] == [
        ('chair', pytest.approx(0.7)), ('sofa', pytest.approx(0.3))]

    from rclpy.parameter import Parameter
    assert node.set_parameters([Parameter('publish_disappeared_objects', value=False)])[0].successful
    node._publish_result(_result([chair, gone]), Header(stamp=stamp_msg(10.0), frame_id='camera'))
    assert [d.id for d in published[-1].detections] == ['3']


def test_map_outputs_are_latched_and_cloud_republishes_only_on_change(node_factory):
    node = node_factory()
    for topic in ('obj_points', 'obj_boxes', 'objects'):
        info, = node.get_publishers_info_by_topic('/' + topic)
        assert info.qos_profile.durability.name == 'TRANSIENT_LOCAL'
        assert info.qos_profile.reliability.name == 'RELIABLE'
        assert info.qos_profile.depth in (0, 1)  # Jazzy's endpoint info does not report depth (0)
    clouds = []
    node.obj_points_pub.publish = clouds.append
    obj = make_object(1, 'chair', [0, 0, 1, 1, 1, 2])
    obj.points_world = np.array([[.1, .2, 1.5]])
    header = Header(stamp=stamp_msg(10.), frame_id='map')
    node._publish_result(_result([obj]), header)
    node._publish_result(_result([obj]), header)
    assert len(clouds) == 1
    obj.points_world = np.array([[.1, .2, 1.6]])
    node._publish_result(_result([obj]), header)
    assert len(clouds) == 2


def test_diagnostics_report_rates_and_mapping_state(node_factory):
    node = _depth_node(node_factory)
    _feed_depth(node, _observation(10.0, 2.0, False))
    spin_until(node, lambda: node._last_result is not None)
    statuses = []
    node._updater.publish = statuses.extend
    node._updater.update()
    by_name = {s.name: s for s in statuses}
    assert {'detector rate', 'mapping rate', 'publish rate', 'mapping state'} <= set(by_name)
    values = {kv.key: kv.value for kv in by_name['mapping state'].values}
    assert values['failures'] == '0' and values['pending_frames'] == '0' and 'grounding_outstanding' in values


# ------------------------------------------------------------ P5 scan ring
def test_scan_ring_keeps_the_last_scans_oldest_first():
    ring = _ScanRing(2)
    ring.append(np.ones((3, 3)))
    ring.append(2 * np.ones((5, 3)))
    ring.append(3 * np.ones((1, 3)))
    assert len(ring) == 2 and ring[0].shape == (5, 3) and ring[1].tolist() == [[3.0, 3.0, 3.0]]
    points = ring.points()
    assert np.isfinite(points).all(axis=1).sum() == 6
    ring.clear()
    assert len(ring) == 0 and not np.isfinite(ring.points()).any()


def test_geometry_only_frames_skip_rgb_decoding(node_factory):
    node = _depth_node(node_factory, '-p', 'detector_rate_hz:=1.0')
    decoded = []
    original = node._decode_rgb
    node._decode_rgb = lambda msg, intr: decoded.append(1) or original(msg, intr)
    for i in range(4):
        _feed_depth(node, _observation(10.0 + 0.1 * i, 2.0, False))
        time.sleep(0.05)
    spin_until(node, lambda: node.pipeline._frame_index == 4)
    assert len(decoded) == 1  # only the detector frame
    rclpy.spin_once(node, timeout_sec=0.0)
