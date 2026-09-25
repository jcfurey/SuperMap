"""Camera-to-dense-cloud labeling with real ROS messages and a fake model."""
import json

import numpy as np
import pytest
import rclpy
from std_msgs.msg import Header

from semantic_mapping.dense_cloud_io import camera_labels_from_dict
from semantic_mapping.dense_cloud_node import DenseCloudMappingNode
from semantic_mapping.dense_yoloe_node import (
    DenseYOLOELabelsNode, annotations_payload, calibrated_frame, mask_runs,
)
from semantic_mapping.ros_msgs import numpy_to_image
from semantic_mapping.types import CameraIntrinsics, Detection2D
from test.ros.helpers import camera_info, stamp_msg
from test.ros.test_dense_cloud_node import cloud


INTRINSICS = CameraIntrinsics(10., 10., 5., 5., 10, 10)


def frame(stamp):
    header = Header(stamp=stamp_msg(stamp), frame_id="map")
    image = numpy_to_image(np.zeros((10, 10, 3), np.uint8), "rgb8", header)
    return image, camera_info(INTRINSICS, header)


class FakeDetector:
    def detect(self, rgb, prompts):
        mask = np.zeros(rgb.shape[:2], bool)
        mask[5, 5] = True
        return [Detection2D(np.array([5., 5., 6., 6.]), "pipe", .9, mask=mask)]


@pytest.fixture
def label_nodes(monkeypatch):
    monkeypatch.setattr(DenseYOLOELabelsNode, "_load_detector",
                        lambda *args: (FakeDetector(), {"name": "YOLOE", "checkpoint_sha256": "fake"}))
    rclpy.init(args=["--ros-args", "-p", "allow_yoloe:=true", "-p", "images_rectified:=true",
                    "-p", "allow_yoloe_labels:=true", "-p", "world_frame:=map", "-p", "min_region_voxels:=1"])
    label = DenseYOLOELabelsNode()
    dense = DenseCloudMappingNode()
    label.messages = []
    label.annotation_pub.publish = label.messages.append
    label.image_pub.get_subscription_count = lambda: 0
    dense.cloud_pub.publish = lambda message: None
    dense.map_pub.publish = lambda message: None
    dense.regions_pub.publish = lambda message: None
    yield label, dense
    label.destroy_node()
    dense.destroy_node()
    rclpy.shutdown()


@pytest.mark.parametrize("mask", [np.zeros((10, 10), bool), np.ones((10, 10), bool), np.eye(10, dtype=bool)])
def test_mask_spans_round_trip_without_changing_membership(mask):
    detection = Detection2D(np.array([0., 0., 10., 10.]), "pipe", .9, mask=mask)
    payload = annotations_payload(frame(1.)[0].header, INTRINSICS, [detection], {})
    with pytest.raises(ValueError, match="explicitly allowed"):
        camera_labels_from_dict(payload, T_world_from_camera=np.eye(4))
    camera = camera_labels_from_dict(payload, T_world_from_camera=np.eye(4), allow_yoloe=True)
    np.testing.assert_array_equal(camera.detections[0].mask, mask)


def test_camera_inference_uses_matching_capture_and_labels_full_cloud(label_nodes):
    labels, dense = label_nodes
    labels._on_camera(*frame(1.))
    labels._on_camera(*frame(1.09))
    message = cloud([[0., 0., 2.], [0., 0., 4.], [10., 0., 2.]], stamp=1.1)
    payload = json.loads(labels.messages[-1].data)
    assert payload["stamp"] == pytest.approx(1.09) and payload["source"] == "yoloe"
    # Exercise inference completing before the dense geometry callback.
    dense._on_annotations(labels.messages[-1])
    dense._on_cloud(message)
    result = dense.pipeline.result()
    assert result.semantic_ids.tolist() == [1, 0, 0]
    assert result.labels[1] == "pipe"
    assert len(result.points_world) == 3 and dense._applied_annotations == 1
    assert dense._last_annotation_model["checkpoint_sha256"] == "fake"


def test_camera_labels_publish_without_any_cloud_or_map(label_nodes):
    labels, _ = label_nodes
    labels._on_camera(*frame(1.))
    labels._on_camera(*frame(1.1))
    assert len(labels.messages) == 2
    assert json.loads(labels.messages[-1].data)["stamp"] == 1.1


def test_dense_map_chooses_nearest_historical_mask_and_keeps_future_labels(label_nodes):
    labels, dense = label_nodes
    for stamp in (1., 1.1, 1.2, 2., 3.):
        labels._on_camera(*frame(stamp))
        dense._on_annotations(labels.messages[-1])
    assert dense._applied_annotations == 0
    dense._on_cloud(cloud([[0., 0., 2.], [10., 0., 2.]], stamp=1.09))
    assert dense.pipeline.result().stats["camera_stamp"] == 1.1
    assert dense._applied_annotations == 1
    assert len(dense._pending_annotations) == 2


def test_future_masks_do_not_require_future_tf_until_their_cloud_arrives(label_nodes):
    labels, dense = label_nodes
    labels._on_camera(*frame(1.))
    payload = json.loads(labels.messages[-1].data)
    payload["frame_id"] = "camera_without_tf_yet"
    from std_msgs.msg import String
    dense._on_annotations(String(data=json.dumps(payload)))
    assert len(dense._pending_annotations) == 1 and dense._rejected_annotations == 0
    dense._on_cloud(cloud([[0., 0., 2.]], stamp=1.))
    assert dense._rejected_annotations == 1
    assert dense.pipeline.result().semantic_ids.tolist() == [0]


@pytest.mark.parametrize("bad", ["distortion", "size", "time", "frame", "rotation"])
def test_camera_calibration_rejects_mismatches(bad):
    image, info = frame(1.)
    if bad == "distortion":
        info.d = [.1, 0., 0., 0., 0.]
    elif bad == "size":
        info.width = 20
    elif bad == "time":
        info.header = Header(stamp=stamp_msg(2.), frame_id="map")
    elif bad == "frame":
        info.header = Header(stamp=stamp_msg(1.), frame_id="wrong")
    else:
        info.r = [0., -1., 0., 1., 0., 0., 0., 0., 1.]
    with pytest.raises(ValueError):
        calibrated_frame(image, info, .2)


def test_missing_local_models_fail_before_inference(tmp_path):
    with pytest.raises(ValueError, match="existing local files"):
        DenseYOLOELabelsNode._load_detector(None, str(tmp_path/'missing.pt'), str(tmp_path/'missing.ts'), 'cpu', .5)
