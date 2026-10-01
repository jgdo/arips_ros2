cmake_minimum_required(VERSION 3.20)
project(arips_semantic_map_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/Point2D.msg"
    "msg/Door.msg"
    "msg/SemanticMap.msg"
    "msg/Segment.msg"
    "msg/SemanticRoute.msg"
    "srv/HandleMapFile.srv"
    "srv/AddDoor.srv"
    "srv/ComputeSemanticRoute.srv"
    DEPENDENCIES std_msgs geometry_msgs nav_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
