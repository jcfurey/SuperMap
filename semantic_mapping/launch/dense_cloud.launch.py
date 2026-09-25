"""Launch full-cloud segmentation without loading any detector or camera model.

Usage:
    ros2 launch semantic_mapping dense_cloud.launch.py
    ros2 launch semantic_mapping dense_cloud.launch.py namespace:=robot1 use_sim_time:=true
    ros2 launch semantic_mapping dense_cloud.launch.py cloud_topic:=ouster/points input_mode:=scan
    ros2 launch semantic_mapping dense_cloud.launch.py autostart:=false   # nav2_lifecycle_manager

Parameters come from ``config`` (a YAML keyed ``/**`` so it applies under any
namespace). Optional launch arguments left empty do not touch the YAML; set
one and it overrides the matching parameter. Topics are relative, so
``namespace`` moves every input and output. TF stays global.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# Launch argument -> node parameter, passed only when the argument is non-empty
# so an unset argument never shadows the YAML.
OPTIONAL_PARAMETERS = ("cloud_topic", "world_frame", "input_mode", "camera_annotations_topic",
                       "manual_annotations_topic")


def _as_bool(text: str) -> bool:
    value = text.strip().lower()
    if value not in ("true", "false", "1", "0", "yes", "no"):
        raise ValueError(f"expected a boolean, got {text!r}")
    return value in ("true", "1", "yes")


def _launch_setup(context):
    def value(name: str) -> str:
        return LaunchConfiguration(name).perform(context).strip()

    overrides = {"use_sim_time": _as_bool(value("use_sim_time")), "autostart": _as_bool(value("autostart"))}
    for name in OPTIONAL_PARAMETERS:
        if value(name):
            overrides[name] = value(name)
    if overrides.get("input_mode", "snapshot") not in ("snapshot", "scan"):
        raise RuntimeError("input_mode must be snapshot or scan")
    return [Node(
        package="semantic_mapping",
        executable="dense_cloud_mapping_node",
        name="dense_cloud_mapping",
        namespace=LaunchConfiguration("namespace"),
        output="screen",
        parameters=[LaunchConfiguration("config"), overrides],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
        remappings=[("/tf", "/tf"), ("/tf_static", "/tf_static")],
    )]


def generate_launch_description() -> LaunchDescription:
    share = FindPackageShare("semantic_mapping")
    arguments = [
        DeclareLaunchArgument("namespace", default_value="", description="Namespace for the node and its topics."),
        DeclareLaunchArgument("use_sim_time", default_value="false", description="Use /clock (bag playback)."),
        DeclareLaunchArgument(
            "autostart", default_value="true",
            description="Configure and activate on startup; false lets nav2_lifecycle_manager drive the node."),
        DeclareLaunchArgument(
            "config", default_value=PathJoinSubstitution([share, "config", "dense_cloud.yaml"]),
            description="Parameter YAML (keyed /** so it applies under any namespace)."),
        DeclareLaunchArgument("log_level", default_value="info", description="Node log level."),
        DeclareLaunchArgument("cloud_topic", default_value="", description="Override cloud_topic (empty: YAML)."),
        DeclareLaunchArgument("world_frame", default_value="", description="Override world_frame (empty: YAML)."),
        DeclareLaunchArgument("input_mode", default_value="", description="Override input_mode: snapshot | scan."),
        DeclareLaunchArgument("camera_annotations_topic", default_value="",
                              description="Override the typed CameraAnnotations input (empty: YAML)."),
        DeclareLaunchArgument("manual_annotations_topic", default_value="",
                              description="Enable the manual JSON annotation input on this topic."),
    ]
    return LaunchDescription(arguments + [OpaqueFunction(function=_launch_setup)])
