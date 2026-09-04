"""rosbag2 round trip: write a bag from the synthetic scene (RGB, CameraInfo, a
LiDAR-frame point cloud, odometry as map -> sensor, static extrinsics on
/tf_static), convert it back with examples/rosbag_to_sequence.py, and check
poses, images, depth, and the resulting detection recall."""
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
import rclpy.serialization
from nav_msgs.msg import Odometry
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage

from semantic_mapping.datasets import SequenceDataset
from semantic_mapping.geometry_utils import (
    back_project_depth, invert_se3, quaternion_to_rotation_matrix, rotation_matrix_to_quaternion, transform_points,
)
from semantic_mapping.ros_msgs import numpy_to_image
from test.ros.helpers import ROOT, camera_info, stamp_msg

rosbag2_py = pytest.importorskip("rosbag2_py")

BASE_T = 1000.0  # keep header stamps away from zero

T_SENSOR_FROM_CAM = np.eye(4)
T_SENSOR_FROM_CAM[:3, :3] = quaternion_to_rotation_matrix(-0.5, 0.5, -0.5, 0.5)
T_SENSOR_FROM_CAM[:3, 3] = [0.05, 0.0, 0.1]
T_SENSOR_FROM_LIDAR = np.eye(4)
T_SENSOR_FROM_LIDAR[:3, 3] = [0.1, 0.0, 0.2]


def _tf(parent, child, T, stamp):
    from geometry_msgs.msg import TransformStamped

    msg = TransformStamped()
    msg.header = Header(stamp=stamp_msg(stamp), frame_id=parent)
    msg.child_frame_id = child
    q = rotation_matrix_to_quaternion(T[:3, :3])
    msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = map(float, T[:3, 3])
    msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w = map(float, q)
    return msg


def _write_bag(dataset, bag_dir):
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    topics = {
        "/camera/color/image_raw": "sensor_msgs/msg/Image", "/camera/color/camera_info": "sensor_msgs/msg/CameraInfo",
        "/lidar/points": "sensor_msgs/msg/PointCloud2", "/odometry": "nav_msgs/msg/Odometry",
        "/tf_static": "tf2_msgs/msg/TFMessage",
    }
    for i, (name, type_name) in enumerate(topics.items()):
        writer.create_topic(rosbag2_py.TopicMetadata(id=i, name=name, type=type_name, serialization_format="cdr"))

    def put(topic, msg, t):
        writer.write(topic, rclpy.serialization.serialize_message(msg), int(round(t * 1e9)))

    intr = dataset.intrinsics
    for i, frame in enumerate(dataset):
        t = BASE_T + frame.stamp
        if i == 0:
            put("/tf_static", TFMessage(transforms=[
                _tf("sensor", "camera_color_optical_frame", T_SENSOR_FROM_CAM, t),
                _tf("sensor", "lidar", T_SENSOR_FROM_LIDAR, t)]), t)
        header = Header(stamp=stamp_msg(t), frame_id="camera_color_optical_frame")
        put("/camera/color/image_raw", numpy_to_image(frame.rgb, "rgb8", header), t)
        put("/camera/color/camera_info", camera_info(intr, header), t)
        T_world_from_sensor = frame.T_world_from_cam @ invert_se3(T_SENSOR_FROM_CAM)
        T_world_from_lidar = T_world_from_sensor @ T_SENSOR_FROM_LIDAR
        pts_world = transform_points(frame.T_world_from_cam, back_project_depth(intr.K, frame.depth))
        pts_lidar = transform_points(invert_se3(T_world_from_lidar), pts_world)
        put("/lidar/points", pc2.create_cloud_xyz32(Header(stamp=stamp_msg(t), frame_id="lidar"), pts_lidar.tolist()), t)
        odom = Odometry(header=Header(stamp=stamp_msg(t), frame_id="map"), child_frame_id="sensor")
        q = rotation_matrix_to_quaternion(T_world_from_sensor[:3, :3])
        p = odom.pose.pose
        p.position.x, p.position.y, p.position.z = map(float, T_world_from_sensor[:3, 3])
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = map(float, q)
        put("/odometry", odom, t)
    writer.close()


def test_bag_round_trip_reproduces_the_sequence_and_its_metrics(dataset, tmp_path):
    bag_dir, out_dir = tmp_path / "synthetic", tmp_path / "seq"
    _write_bag(dataset, bag_dir)

    subprocess.run([sys.executable, str(ROOT / "examples" / "rosbag_to_sequence.py"), str(bag_dir), "--out_dir", str(out_dir),
                    "--pointcloud_topic", "/lidar/points", "--odometry_topic", "/odometry",
                    "--world_frame", "map", "--camera_frame", "camera_color_optical_frame"],
                   check=True, capture_output=True, text=True)
    info = json.loads((out_dir / "sequence_info.json").read_text())
    assert info["frames"] == len(dataset), info

    converted = SequenceDataset(out_dir)
    for a, b in zip(dataset, converted):
        assert np.allclose(a.T_world_from_cam, b.T_world_from_cam, atol=1e-5)
        assert np.array_equal(a.rgb, b.rgb)
        valid = (a.depth > 0) & (b.depth > 0)
        assert valid.mean() > 0.95 and np.abs(a.depth[valid] - b.depth[valid]).max() < 0.02

    shutil.copytree(dataset.detections_dir, out_dir / "detections")
    shutil.copy(dataset.data_dir / "scene_ground_truth.json", out_dir / "scene_ground_truth.json")
    output = subprocess.run([sys.executable, str(ROOT / "examples" / "evaluate.py"), "--data_dir", str(out_dir),
                             "--no_segmentation", "--config", str(ROOT / "config" / "semantic_mapping.yaml"),
                             "--prompts", str(ROOT / "config" / "prompts.yaml")],
                            check=True, capture_output=True, text=True).stdout
    line = next(l for l in output.splitlines() if l.startswith("mean detection recall"))
    assert float(line.split(":")[1]) > 0.9, output
