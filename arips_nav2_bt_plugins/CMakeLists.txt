cmake_minimum_required(VERSION 3.20)
project(arips_nav2_bt_plugins)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(arips_action_msgs REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(arips_semantic_map_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(nav2_behavior_tree REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(nav2_ros_common REQUIRED)
find_package(rclcpp REQUIRED)

function(add_bt_plugin target source)
  add_library(${target} SHARED ${source})
  target_compile_definitions(${target} PRIVATE BT_PLUGIN_EXPORT)
  target_include_directories(${target} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(${target}
    PUBLIC
    behaviortree_cpp::behaviortree_cpp
    geometry_msgs::geometry_msgs__rosidl_typesupport_cpp
    nav2_msgs::nav2_msgs__rosidl_typesupport_cpp
    rclcpp::rclcpp
  )
endfunction()

add_bt_plugin(arips_nav2_bt_plugins_route_counter src/route_counter.cpp)
add_bt_plugin(arips_nav2_bt_plugins_compute_semantic_route src/compute_semantic_route.cpp)
target_link_libraries(
  arips_nav2_bt_plugins_compute_semantic_route
  PUBLIC
    arips_semantic_map_msgs::arips_semantic_map_msgs__rosidl_typesupport_cpp
    nav2_ros_common::nav2_ros_common)
target_link_libraries(
  arips_nav2_bt_plugins_compute_semantic_route
  PUBLIC nav2_behavior_tree::nav2_behavior_tree)
add_bt_plugin(arips_nav2_bt_plugins_cross_door_step src/cross_door_step.cpp)
target_link_libraries(
  arips_nav2_bt_plugins_cross_door_step
  PUBLIC
    arips_action_msgs::arips_action_msgs__rosidl_typesupport_cpp
    nlohmann_json::nlohmann_json
    nav2_behavior_tree::nav2_behavior_tree
    nav2_ros_common::nav2_ros_common)
add_bt_plugin(
  arips_nav2_bt_plugins_pop_next_semantic_route_segment
  src/pop_next_semantic_route_segment.cpp)
target_link_libraries(
  arips_nav2_bt_plugins_pop_next_semantic_route_segment
  PUBLIC arips_semantic_map_msgs::arips_semantic_map_msgs__rosidl_typesupport_cpp)

install(
  TARGETS
    arips_nav2_bt_plugins_route_counter
    arips_nav2_bt_plugins_compute_semantic_route
    arips_nav2_bt_plugins_cross_door_step
    arips_nav2_bt_plugins_pop_next_semantic_route_segment
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

ament_export_dependencies(arips_action_msgs behaviortree_cpp arips_semantic_map_msgs geometry_msgs nlohmann_json nav2_behavior_tree nav2_msgs nav2_ros_common rclcpp)
ament_package()