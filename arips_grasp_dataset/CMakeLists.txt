cmake_minimum_required(VERSION 3.20)
project(arips_grasp_dataset)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(arips_action_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_storage REQUIRED)

# Bag recorder (C++: serialized messages, no frames skipped)
add_executable(grasp_bag_recorder src/grasp_bag_recorder.cpp)
target_link_libraries(grasp_bag_recorder
  ${arips_action_msgs_TARGETS}
  rclcpp::rclcpp
  rosbag2_cpp::rosbag2_cpp
  rosbag2_storage::rosbag2_storage
)
install(TARGETS grasp_bag_recorder DESTINATION lib/${PROJECT_NAME})

# Episode orchestration (Python)
ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS scripts/record_grasp_dataset DESTINATION lib/${PROJECT_NAME})

ament_package()
