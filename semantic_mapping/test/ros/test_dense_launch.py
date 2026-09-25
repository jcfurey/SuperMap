"""The dense launch files load, and every YAML key is a declared parameter with the YAML value."""
import importlib.util
from pathlib import Path

import pytest
import rclpy
import yaml

from semantic_mapping.dense_cloud_node import DenseCloudMappingNode
from semantic_mapping.dense_yoloe_node import DenseYOLOELabelsNode

PACKAGE = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("config, factory", [("dense_cloud.yaml", DenseCloudMappingNode),
                                             ("dense_yoloe.yaml", DenseYOLOELabelsNode)])
def test_yaml_matches_declared_parameters_under_a_namespace(config, factory):
    path = PACKAGE / "config" / config
    params = yaml.safe_load(path.read_text())["/**"]["ros__parameters"]
    rclpy.init(args=["--ros-args", "--params-file", str(path), "-p", "autostart:=false", "-r", "__ns:=/robot1"])
    try:
        node = factory()
        assert node.get_namespace() == "/robot1"
        for name, value in params.items():
            assert node.has_parameter(name), f"{config}: {name} is not declared"
            actual = node.get_parameter(name).value
            assert (list(actual) if isinstance(actual, (list, tuple)) else actual) == value, name
        for name, value in params.items():
            if name.endswith("_topic") and value:
                assert not value.startswith("/"), f"{name} should be relative so namespaces apply"
        node.destroy_node()
    finally:
        rclpy.try_shutdown()


@pytest.mark.parametrize("launch_file", ["dense_cloud.launch.py", "dense_yoloe.launch.py"])
def test_launch_files_declare_namespace_and_sim_time(launch_file):
    spec = importlib.util.spec_from_file_location("launch_under_test", PACKAGE / "launch" / launch_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    description = module.generate_launch_description()
    names = {argument.name for argument in description.get_launch_arguments()}
    assert {"namespace", "use_sim_time", "autostart", "config", "log_level"} <= names
