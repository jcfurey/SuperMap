"""Launch the live SuperMap semantic mapping node.

Usage:
    ros2 launch semantic_mapping semantic_mapping.launch.py
    ros2 launch semantic_mapping semantic_mapping.launch.py config:=/path/to/custom.yaml
    ros2 launch semantic_mapping semantic_mapping.launch.py namespace:=robot1 use_sim_time:=true
    ros2 launch semantic_mapping semantic_mapping.launch.py autostart:=false   # nav2_lifecycle_manager

Parameters come from ``config`` (a YAML keyed ``/**`` so it applies under any
namespace). Optional launch arguments left empty do not touch the YAML; set
one and it overrides the matching parameter. Topics are relative, so
``namespace`` moves every input and output; remap individual topics with
``--ros-args -r`` on a custom launch, or set the ``*_topic`` parameters.

The node resolves world_frame -> camera_frame (Eq. 3) and the incoming point
cloud's frame -> camera_frame through TF2, so a sensor_frame -> camera_frame
extrinsic must be present in the TF tree -- normally from a URDF +
robot_state_publisher or the camera driver. Only when neither provides it,
set ``publish_static_camera_tf:=true`` together with the calibrated
``camera_x ... camera_qw``; there is no default extrinsic, because a
placeholder would silently mis-calibrate the camera.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

EXTRINSIC_ARGS = ("camera_x", "camera_y", "camera_z", "camera_qx", "camera_qy", "camera_qz", "camera_qw")

# Launch argument -> node parameter, passed only when the argument is non-empty
# so an unset argument never shadows the YAML.
OPTIONAL_PARAMETERS = {
    "prompts_file": ("prompts_file", str),
    "camera_frame": ("camera_frame", str),
    "world_frame": ("world_frame", str),
    "map_load_path": ("map_load_path", str),
    "map_save_path": ("map_save_path", str),
    "detector": ("detector", str),
}


def _as_bool(text: str) -> bool:
    value = text.strip().lower()
    if value not in ("true", "false", "1", "0", "yes", "no"):
        raise ValueError(f"expected a boolean, got {text!r}")
    return value in ("true", "1", "yes")


def _launch_setup(context):
    def value(name: str) -> str:
        return LaunchConfiguration(name).perform(context).strip()

    overrides = {
        "use_sim_time": _as_bool(value("use_sim_time")),
        "autostart": _as_bool(value("autostart")),
    }
    for arg, (parameter, cast) in OPTIONAL_PARAMETERS.items():
        if value(arg):
            overrides[parameter] = cast(value(arg))

    actions = []
    if _as_bool(value("publish_static_camera_tf")):
        missing = [name for name in EXTRINSIC_ARGS if not value(name)]
        if missing or not value("sensor_frame") or not value("camera_frame"):
            raise RuntimeError(
                "publish_static_camera_tf:=true needs the calibrated extrinsic: set sensor_frame, camera_frame "
                f"and {', '.join(EXTRINSIC_ARGS)} (missing: {', '.join(missing) or 'frames'})")
        arguments = []
        for name in EXTRINSIC_ARGS:
            arguments += [f"--{name.removeprefix('camera_')}", value(name)]
        arguments += ["--frame-id", value("sensor_frame"), "--child-frame-id", value("camera_frame")]
        actions.append(Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="camera_extrinsics_tf",
            namespace=LaunchConfiguration("namespace"),
            output="screen",
            arguments=arguments,
            parameters=[{"use_sim_time": overrides["use_sim_time"]}],
        ))

    actions.append(Node(
        package="semantic_mapping",
        executable="semantic_mapping_node",
        name="semantic_mapping_node",
        namespace=LaunchConfiguration("namespace"),
        output="screen",
        parameters=[LaunchConfiguration("config"), overrides],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
        # Keep TF global under a namespace, as Nav2 does.
        remappings=[("/tf", "/tf"), ("/tf_static", "/tf_static")],
    ))
    return actions


def generate_launch_description() -> LaunchDescription:
    share = FindPackageShare("semantic_mapping")
    arguments = [
        DeclareLaunchArgument("namespace", default_value="", description="Namespace for the node and its topics."),
        DeclareLaunchArgument("use_sim_time", default_value="false", description="Use /clock (bag playback)."),
        DeclareLaunchArgument(
            "autostart", default_value="true",
            description="Configure and activate on startup; false lets nav2_lifecycle_manager drive the node."),
        DeclareLaunchArgument(
            "config", default_value=PathJoinSubstitution([share, "config", "semantic_mapping.yaml"]),
            description="Parameter YAML (keyed /** so it applies under any namespace)."),
        DeclareLaunchArgument("log_level", default_value="info", description="Node log level."),
        DeclareLaunchArgument(
            "prompts_file", default_value="",
            description="Override prompts_file (empty: YAML value; relative paths resolve in the package share)."),
        DeclareLaunchArgument("detector", default_value="", description="Override detector (empty: YAML value)."),
        DeclareLaunchArgument("world_frame", default_value="", description="Override world_frame (empty: YAML)."),
        DeclareLaunchArgument(
            "camera_frame", default_value="",
            description="Override camera_frame (empty: YAML). Also the child frame of the optional static TF."),
        DeclareLaunchArgument(
            "map_load_path", default_value="",
            description="Override map_load_path: saved map restored on configure (empty: YAML value)."),
        DeclareLaunchArgument(
            "map_save_path", default_value="",
            description="Override map_save_path: default ~/save_map and autosave directory (empty: YAML value)."),
        DeclareLaunchArgument(
            "publish_static_camera_tf", default_value="false",
            description="Publish sensor_frame -> camera_frame from camera_x ... camera_qw. Only for setups "
                        "without URDF/driver TF; all extrinsic arguments are then required."),
        DeclareLaunchArgument(
            "sensor_frame", default_value="",
            description="Parent frame of the optional static camera extrinsic (e.g. the LiDAR/body frame)."),
        *[DeclareLaunchArgument(name, default_value="", description=f"Calibrated extrinsic, {name[7:]}.")
          for name in EXTRINSIC_ARGS],
    ]
    return LaunchDescription([*arguments, OpaqueFunction(function=_launch_setup)])
