"""robot_description-based platform masking in the object node and the dense-cloud node."""
import json
import time

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Header, String

from semantic_mapping.dense_cloud_node import DenseCloudMappingNode
from semantic_mapping.detectors.base import Detector
from semantic_mapping.types import Detection2D
from test.ros.helpers import feed_frame, spin_until
from test.ros.test_dense_cloud_node import BASE_ARGS, capture_publishers, cloud

# A thin plate 1 m ahead of the camera covering everything below the optical
# axis (optical y points down): the robot's hood in the lower half of the view.
HOOD_URDF = """<robot name='dozer'>
  <link name='base_link'/>
  <link name='hood'><visual><origin xyz='0 2.5 0'/><geometry><box size='20 5 0.05'/></geometry></visual></link>
</robot>"""


def set_link_tf(tf_buffer, parent: str, child: str = "hood", z: float = 1.0) -> None:
    tf = TransformStamped()
    tf.header = Header(frame_id=parent)
    tf.child_frame_id = child
    tf.transform.translation.z = z
    tf.transform.rotation.w = 1.0
    tf_buffer.set_transform_static(tf, "test")


class FixedDetector(Detector):
    """Two detections: one on the hood (lower half), one above it."""

    def detect(self, rgb_image, prompts=None, **kwargs):
        height, width = rgb_image.shape[:2]
        on_hood = np.zeros((height, width), bool)
        on_hood[height * 3 // 4:, :width // 2] = True
        above = np.zeros((height, width), bool)
        above[height // 8:height // 3, width // 2:] = True
        return [Detection2D(np.array([0., height * 3 // 4, width // 2, height]), "bulldozer", 0.9, mask=on_hood),
                Detection2D(np.array([width // 2, height // 8, width, height // 3]), "person", 0.9, mask=above)]


def masked_node(node_factory, *extra):
    node = node_factory("-p", "platform_mask.enabled:=true", "-p", "platform_mask.padding_px:=0", *extra)
    node.detector = FixedDetector()
    observations = []
    process = node.pipeline.process_frame
    node.pipeline.process_frame = lambda observation: observations.append(observation) or process(observation)
    return node, observations


def test_object_node_drops_detections_on_the_robot_and_hides_depth_behind_it(node_factory, dataset):
    node, observations = masked_node(node_factory)
    node._platform._on_description(String(data=HOOD_URDF))
    set_link_tf(node.tf_buffer, node.camera_frame)
    frame = next(iter(dataset))
    feed_frame(node, dataset, frame)
    spin_until(node, lambda: observations and observations[-1].detections_evaluated)

    observation = observations[-1]
    assert [d.label for d in observation.detections] == ["person"]
    assert node._counters["platform_detections"] == 1
    half = observation.intrinsics.height // 2
    assert not observation.depth[half + 2:].any(), "returns behind the hood are unknown depth"
    assert observation.depth[:half - 2].any()


def test_frames_wait_for_the_description_then_are_skipped(node_factory, dataset):
    node, observations = masked_node(node_factory, "-p", "tf_wait_sec:=0.0")
    frames = list(dataset)[:2]
    feed_frame(node, dataset, frames[0])
    assert node._counters["platform_unavailable"] == 1 and not node._pending_frames
    node._platform._on_description(String(data=HOOD_URDF))
    feed_frame(node, dataset, frames[1])
    assert node._counters["tf_failures"] == 1, "the hood link has no TF yet"
    assert not observations


def test_description_arrives_on_the_latched_topic(node_factory):
    node, _ = masked_node(node_factory, "-p", "platform_mask.robot_description_topic:=/test_platform/robot_description")
    publisher = node.create_publisher(String, "/test_platform/robot_description", QoSProfile(
        depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    publisher.publish(String(data=HOOD_URDF))
    deadline = time.monotonic() + 5.0
    while node._platform.model is None and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    assert node._platform.model is not None and node._platform.model.links == ["hood"]
    # An unusable update keeps the working model.
    node._platform._on_description(String(data="<robot name='empty'/>"))
    assert node._platform.model.links == ["hood"]


@pytest.fixture
def masked_dense_node():
    rclpy.init(args=[*BASE_ARGS, "-p", "platform_mask.enabled:=true", "-p", "platform_mask.padding_px:=0"])
    node = DenseCloudMappingNode()
    node.autostart_now()
    capture_publishers(node)
    yield node
    node.destroy_node()
    rclpy.try_shutdown()


def test_dense_node_never_labels_through_the_robot(masked_dense_node):
    node = masked_dense_node
    node._platform._on_description(String(data=HOOD_URDF))
    set_link_tf(node.tf_buffer, "map")
    # The camera is the map frame: one point above the optical axis, one below it (behind the hood).
    node._on_cloud(cloud([[0., -1., 2.], [0., .8, 2.]]))
    payload = {"source": "manual", "rectified": True, "stamp": 1., "frame_id": "map",
               "intrinsics": {"fx": 10., "fy": 10., "cx": 5., "cy": 5., "width": 10, "height": 10},
               "detections": [{"label": "person", "mask_indices": [5]},
                              {"label": "bulldozer", "mask_indices": [95]}]}
    node._on_annotations(String(data=json.dumps(payload)))
    result = node.pipeline.result()
    assert [result.labels[i] for i in result.semantic_ids] == ["person", "unknown"]
    assert node._stats.get("platform_detections") == 1 and node._stats.get("annotations_applied") == 1
