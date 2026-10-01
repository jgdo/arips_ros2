cmake_minimum_required(VERSION 3.20)
project(arips_nav2_bt_plugins)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
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
add_bt_plugin(arips_nav2_bt_plugins_pop_next_edge src/pop_next_edge.cpp)

install(
  TARGETS
    arips_nav2_bt_plugins_route_counter
    arips_nav2_bt_plugins_pop_next_edge
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

ament_export_dependencies(behaviortree_cpp geometry_msgs nav2_msgs rclcpp)
ament_package()