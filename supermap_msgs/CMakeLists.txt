cmake_minimum_required(VERSION 3.16)
project(supermap_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(action_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MaskDetection.msg"
  "msg/CameraAnnotations.msg"
  "msg/Region.msg"
  "msg/RegionArray.msg"
  "srv/SaveMap.srv"
  "srv/LoadMap.srv"
  "action/GroundInstruction.action"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs nav_msgs vision_msgs action_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
