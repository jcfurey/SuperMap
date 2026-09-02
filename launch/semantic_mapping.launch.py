"""Launch the live SuperMap semantic mapping node.

Usage:
    ros2 launch semantic_mapping semantic_mapping.launch.py
    ros2 launch semantic_mapping semantic_mapping.launch.py config:=/path/to/custom.yaml
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    config_arg = DeclareLaunchArgument(
        "config",
        default_value=[FindPackageShare("semantic_mapping"), "/config/semantic_mapping.yaml"],
        description="Path to the semantic_mapping parameter YAML file.",
    )
    prompts_arg = DeclareLaunchArgument(
        "prompts_file",
        default_value=[FindPackageShare("semantic_mapping"), "/config/prompts.yaml"],
        description="Path to the open-vocabulary detection prompt list.",
    )
    node = Node(
        package="semantic_mapping",
        executable="semantic_mapping_node",
        name="semantic_mapping_node",
        output="screen",
        parameters=[
            LaunchConfiguration("config"),
            {"prompts_file": LaunchConfiguration("prompts_file")},
        ],
    )

    return LaunchDescription([config_arg, prompts_arg, node])
