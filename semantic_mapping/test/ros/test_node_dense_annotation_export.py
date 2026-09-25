"""Sharing existing inference must preserve the object tracker and box publishers."""
from types import SimpleNamespace

import numpy as np
import pytest
from std_msgs.msg import Header
from supermap_msgs.msg import CameraAnnotations

from semantic_mapping.dense_cloud_io import camera_labels_from_msg
from semantic_mapping.node import _PendingFrame
from test.ros.helpers import camera_info, stamp_msg
from test.test_pipeline import _observation


def test_existing_tracker_exports_masks_and_still_publishes_boxes(node_factory):
    node = node_factory('-p', 'min_hits_to_confirm:=1', '-p', 'publish_rate_hz:=100.0')
    masks, boxes, points = [], [], []
    node.dense_annotations_pub = SimpleNamespace(publish=masks.append)
    node._dense_annotation_model = {'name': 'YOLOE', 'checkpoint_sha256': 'test-only'}
    node.obj_boxes_pub.publish = boxes.append
    node.obj_points_pub.publish = points.append
    observation = _observation(1., 2., True)
    for detection in observation.detections:
        detection.mask = np.zeros(observation.depth.shape, bool)
        x1, y1, x2, y2 = map(int, detection.bbox)
        detection.mask[y1:y2, x1:x2] = True
    observation.detections_evaluated = True
    header = Header(stamp=stamp_msg(1.), frame_id=node.camera_frame)
    pending = _PendingFrame(observation, header, annotate=True,
                            camera_info=camera_info(observation.intrinsics, header))
    with node._lock:
        node._pending_frames.append(pending)
        node._pending_by_id[observation.frame_id] = pending
        node._flush_pending_frames()
    assert len(masks) == 1 and boxes and points and points[-1].width > 0
    assert boxes[-1].markers and node.pipeline.object_map.objects
    msg = masks[0]
    assert isinstance(msg, CameraAnnotations)
    assert msg.source == 'yoloe' and msg.rectified and msg.header.stamp.sec == 1
    assert msg.header.frame_id == node.camera_frame and 'YOLOE' in msg.model
    decoded = camera_labels_from_msg(msg, T_world_from_camera=np.eye(4), allow_yoloe=True)
    np.testing.assert_array_equal(decoded.detections[0].mask, observation.detections[0].mask)


def test_export_policy_requires_yoloe_rectification_and_local_assets(node_factory, tmp_path):
    base = ('-p', 'publish_dense_camera_annotations:=true')
    node = node_factory(*base, activate=False)
    with pytest.raises(ValueError, match='YOLOE exception'):
        node._configure_dense_annotations()
    node = node_factory(*base, '-p', 'detector:=yoloe', activate=False)
    with pytest.raises(ValueError, match='rectified'):
        node._configure_dense_annotations()
    rectified = (*base, '-p', 'detector:=yoloe', '-p', 'camera_images_rectified:=true')
    node = node_factory(*rectified, activate=False)
    with pytest.raises(ValueError, match='existing local'):
        node._configure_dense_annotations()
    checkpoint, encoder = tmp_path / 'model.pt', tmp_path / 'encoder.ts'
    checkpoint.write_bytes(b'fixture')
    encoder.write_bytes(b'fixture')
    assets = ('-p', f'yoloe.checkpoint:={checkpoint}', '-p', f'yoloe.text_encoder_path:={encoder}')
    node = node_factory(*rectified, *assets, activate=False)
    assert len(node._configure_dense_annotations()['checkpoint_sha256']) == 64
    # The deprecated dense-only flag still enables rectified input.
    node = node_factory(*base, '-p', 'detector:=yoloe', '-p', 'dense_camera_images_rectified:=true', *assets,
                        activate=False)
    assert node._configure_dense_annotations()['name'] == 'YOLOE'
    node = node_factory(*rectified, *assets, '-p', 'yoloe.sam2_checkpoint:=extra-model.pt', activate=False)
    with pytest.raises(ValueError, match='extra models'):
        node._configure_dense_annotations()
