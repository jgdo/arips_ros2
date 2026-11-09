cmake_minimum_required(VERSION 3.20)
project(arips_action_msgs)

find_package(ament_cmake REQUIRED)
find_package(action_msgs REQUIRED)
find_package(arips_semantic_map_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
    "action/CrossDoorStep.action"
    "action/GrabDoorHandle.action"
    "srv/GetDoorGraspingApproachPose.srv"
    "srv/GetDoorGraspingApproachPoseByIndex.srv"
    DEPENDENCIES action_msgs arips_semantic_map_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()