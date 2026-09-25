"""Launch local YOLOE mask labelling for the dense cloud (explicit policy exception).

Usage:
    ros2 launch semantic_mapping dense_yoloe.launch.py allow_yoloe:=true images_rectified:=true \\
        checkpoint:=/models/yoloe-v8l-seg.pt text_encoder_path:=/models/mobileclip_blt.ts
    ros2 launch semantic_mapping dense_yoloe.launch.py ... with_dense_cloud:=true use_sim_time:=true

Parameters come from ``config`` (``config/dense_yoloe.yaml``, keyed ``/**``).
Optional launch arguments left empty do not touch the YAML. The node publishes
``supermap_msgs/CameraAnnotations`` on a relative topic, so a namespace moves
it together with ``dense_cloud_mapping``; ``with_dense_cloud:=true`` also
starts the cloud node with YOLOE labels allowed.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

OPTIONAL_STRINGS = ("checkpoint", "text_encoder_path", "device", "rgb_topic", "camera_info_topic")
OPTIONAL_BOOLS = ("allow_yoloe", "images_rectified")


def _as_bool(text: str) -> bool:
    value = text.strip().lower()
    if value not in ("true", "false", "1", "0", "yes", "no"):
        raise ValueError(f"expected a boolean, got {text!r}")
    return value in ("true", "1", "yes")


def _launch_setup(context):
    def value(name: str) -> str:
        return LaunchConfiguration(name).perform(context).strip()

    common = {"use_sim_time": _as_bool(value("use_sim_time")), "autostart": _as_bool(value("autostart"))}
    overrides = dict(common)
    for name in OPTIONAL_STRINGS:
        if value(name):
            overrides[name] = value(name)
    for name in OPTIONAL_BOOLS:
        if value(name):
            overrides[name] = _as_bool(value(name))
    log_level = ["--ros-args", "--log-level", LaunchConfiguration("log_level")]
    actions = [Node(
        package="semantic_mapping",
        executable="dense_yoloe_labels_node",
        name="dense_yoloe_labels",
        namespace=LaunchConfiguration("namespace"),
        output="screen",
        parameters=[LaunchConfiguration("config"), overrides],
        arguments=log_level,
    )]
    if _as_bool(value("with_dense_cloud")):
        actions.append(Node(
            package="semantic_mapping",
            executable="dense_cloud_mapping_node",
            name="dense_cloud_mapping",
            namespace=LaunchConfiguration("namespace"),
            output="screen",
            parameters=[LaunchConfiguration("dense_cloud_config"), {**common, "allow_yoloe_labels": True}],
            arguments=log_level,
            remappings=[("/tf", "/tf"), ("/tf_static", "/tf_static")],
        ))
    return actions


def generate_launch_description() -> LaunchDescription:
    share = FindPackageShare("semantic_mapping")
    arguments = [
        DeclareLaunchArgument("namespace", default_value="", description="Namespace for the nodes and topics."),
        DeclareLaunchArgument("use_sim_time", default_value="false", description="Use /clock (bag playback)."),
        DeclareLaunchArgument(
            "autostart", default_value="true",
            description="Configure (load + warm up the model) and activate on startup; "
                        "false lets nav2_lifecycle_manager drive the node."),
        DeclareLaunchArgument(
            "config", default_value=PathJoinSubstitution([share, "config", "dense_yoloe.yaml"]),
            description="Parameter YAML (keyed /** so it applies under any namespace)."),
        DeclareLaunchArgument(
            "dense_cloud_config", default_value=PathJoinSubstitution([share, "config", "dense_cloud.yaml"]),
            description="Parameter YAML for dense_cloud_mapping when with_dense_cloud:=true."),
        DeclareLaunchArgument("with_dense_cloud", default_value="false",
                              description="Also start dense_cloud_mapping with YOLOE labels allowed."),
        DeclareLaunchArgument("log_level", default_value="info", description="Node log level."),
        DeclareLaunchArgument("allow_yoloe", default_value="", description="Override allow_yoloe (empty: YAML)."),
        DeclareLaunchArgument("images_rectified", default_value="",
                              description="Override images_rectified (empty: YAML)."),
        DeclareLaunchArgument("checkpoint", default_value="", description="Local YOLOE -seg checkpoint."),
        DeclareLaunchArgument("text_encoder_path", default_value="", description="Local MobileCLIP TorchScript."),
        DeclareLaunchArgument("device", default_value="", description="Override device (empty: YAML)."),
        DeclareLaunchArgument("rgb_topic", default_value="", description="Override the rectified RGB topic."),
        DeclareLaunchArgument("camera_info_topic", default_value="", description="Override the CameraInfo topic."),
    ]
    return LaunchDescription(arguments + [OpaqueFunction(function=_launch_setup)])
