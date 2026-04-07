cmake_minimum_required(VERSION 3.8)
project(component_manager_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/StartComponent.srv"
  "srv/StopComponent.srv"
  "srv/ListComponents.srv"
)

ament_package()
