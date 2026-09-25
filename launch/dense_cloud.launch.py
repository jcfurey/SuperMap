"""Launch full-cloud segmentation without loading any detector or camera model."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=[FindPackageShare("semantic_mapping"), "/config/dense_cloud.yaml"]),
        DeclareLaunchArgument("cloud_topic", default_value="/points"),
        DeclareLaunchArgument("world_frame", default_value="map"),
        DeclareLaunchArgument("input_mode", default_value="snapshot", choices=["snapshot", "scan"]),
        Node(package="semantic_mapping", executable="dense_cloud_mapping_node", name="dense_cloud_mapping",
             output="screen", parameters=[LaunchConfiguration("config"), {
                 "cloud_topic": LaunchConfiguration("cloud_topic"),
                 "world_frame": LaunchConfiguration("world_frame"),
                 "input_mode": LaunchConfiguration("input_mode"),
             }]),
    ])
