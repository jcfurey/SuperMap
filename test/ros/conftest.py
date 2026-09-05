"""End-to-end checks of the ROS 2 node and the bag converter.

These need a sourced ROS 2 environment (rclpy, sensor_msgs_py, tf2_ros,
rosbag2_py). Without one the test modules in this directory are not
collected at all (``collect_ignore_glob`` below); a skip raised while
importing this conftest would make pytest 7 skip the entire ``test``
package, which is what CI runs. The Docker CI job runs them inside the
Jazzy image. Frames of the synthetic scene are turned into real messages
and handed to the node's synchronized callback directly; executor-driven
tests also check result delivery and deadlines when input or /clock stops.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HAS_ROS = importlib.util.find_spec("rclpy") is not None
if not HAS_ROS:
    collect_ignore_glob = ["test_*.py"]


@pytest.fixture(scope="session")
def scene_dir(tmp_path_factory) -> Path:
    """The synthetic scene: the checked-out one if it was generated, else a fresh one in a temp dir."""
    existing = ROOT / "data" / "example_scene"
    if (existing / "intrinsics.json").exists():
        return existing
    out = tmp_path_factory.mktemp("scene") / "example_scene"
    subprocess.run(
        [sys.executable, str(ROOT / "examples" / "prepare_example_dataset.py"), "--out_dir", str(out)],
        check=True, capture_output=True, text=True,
    )
    return out


@pytest.fixture(scope="session")
def dataset(scene_dir):
    from semantic_mapping.datasets import SequenceDataset

    return SequenceDataset(scene_dir)


@pytest.fixture
def node_factory(dataset):
    """``make(*ros_args)`` builds a node with the offline detector replaying the scene's detections.

    Parameters are given as ``-p name:=value`` pairs. Making a second node in
    the same test tears the first one down, since parameters travel through
    ``rclpy.init``. Publishers are silenced unless ``silence_publishers=False``.
    """
    import rclpy

    from semantic_mapping.node import SemanticMappingNode

    nodes: list = []

    def make(*ros_args: str, silence_publishers: bool = True):
        if rclpy.ok():
            for node in nodes:
                node.destroy_node()
            nodes.clear()
            rclpy.shutdown()
        rclpy.init(args=[
            "--ros-args", "-p", "detector:=offline", "-p", f"offline.detections_dir:={dataset.detections_dir}",
            "-p", f"prompts_file:={ROOT / 'config' / 'prompts.yaml'}", "-p", "detector_rate_hz:=100.0",
            *ros_args,
        ])
        node = SemanticMappingNode()
        if silence_publishers:
            for publisher in (node.obj_boxes_pub, node.obj_points_pub, node.annotated_image_pub):
                publisher.publish = lambda message: None
        nodes.append(node)
        return node

    yield make
    for node in nodes:
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
