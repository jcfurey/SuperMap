"""Launch the live SuperMap semantic mapping node.

Usage:
    ros2 launch semantic_mapping semantic_mapping.launch.py
    ros2 launch semantic_mapping semantic_mapping.launch.py config:=/path/to/custom.yaml

The node resolves world_frame -> camera_frame (Eq. 3) and the incoming point
cloud's frame -> camera_frame through TF2, so a sensor_frame -> camera_frame
extrinsic must be present in the TF tree. If your setup doesn't already
publish it (via a URDF + robot_state_publisher), this launch file publishes
a fixed one with static_transform_publisher -- override camera_x/.../camera_qw
to your actual calibration, or set publish_static_camera_tf:=false if a URDF
already covers it.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
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

    sensor_frame_arg = DeclareLaunchArgument(
        "sensor_frame", default_value="sensor",
        description="TF frame the upstream SLAM backbone's dynamic world_frame transform targets "
                     "(SuperOdometry's sensor_frame parameter defaults to 'sensor').",
    )
    camera_frame_arg = DeclareLaunchArgument(
        "camera_frame", default_value="camera_color_optical_frame",
        description="TF frame of the RGB camera's optical center; must match config's camera_frame.",
    )
    publish_static_camera_tf_arg = DeclareLaunchArgument(
        "publish_static_camera_tf", default_value="true",
        description="Publish the sensor_frame -> camera_frame extrinsic below via "
                     "static_transform_publisher. Set to false if a URDF / robot_state_publisher "
                     "already provides this transform.",
    )
    # Default rotates LiDAR/body axes (x-forward, z-up) into the camera optical
    # convention (z-forward, y-down) with zero translation; replace with your
    # actual calibrated extrinsic.
    camera_x_arg = DeclareLaunchArgument("camera_x", default_value="0.0")
    camera_y_arg = DeclareLaunchArgument("camera_y", default_value="0.0")
    camera_z_arg = DeclareLaunchArgument("camera_z", default_value="0.0")
    camera_qx_arg = DeclareLaunchArgument("camera_qx", default_value="-0.5")
    camera_qy_arg = DeclareLaunchArgument("camera_qy", default_value="0.5")
    camera_qz_arg = DeclareLaunchArgument("camera_qz", default_value="-0.5")
    camera_qw_arg = DeclareLaunchArgument("camera_qw", default_value="0.5")

    static_camera_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_extrinsics_tf",
        output="screen",
        arguments=[
            "--x", LaunchConfiguration("camera_x"),
            "--y", LaunchConfiguration("camera_y"),
            "--z", LaunchConfiguration("camera_z"),
            "--qx", LaunchConfiguration("camera_qx"),
            "--qy", LaunchConfiguration("camera_qy"),
            "--qz", LaunchConfiguration("camera_qz"),
            "--qw", LaunchConfiguration("camera_qw"),
            "--frame-id", LaunchConfiguration("sensor_frame"),
            "--child-frame-id", LaunchConfiguration("camera_frame"),
        ],
        condition=IfCondition(LaunchConfiguration("publish_static_camera_tf")),
    )

    node = Node(
        package="semantic_mapping",
        executable="semantic_mapping_node",
        name="semantic_mapping_node",
        output="screen",
        parameters=[
            LaunchConfiguration("config"),
            {
                "prompts_file": LaunchConfiguration("prompts_file"),
                "camera_frame": LaunchConfiguration("camera_frame"),
            },
        ],
    )

    return LaunchDescription([
        config_arg, prompts_arg,
        sensor_frame_arg, camera_frame_arg, publish_static_camera_tf_arg,
        camera_x_arg, camera_y_arg, camera_z_arg,
        camera_qx_arg, camera_qy_arg, camera_qz_arg, camera_qw_arg,
        static_camera_tf_node, node,
    ])
