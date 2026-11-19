cmake_minimum_required(VERSION 3.20)
project(arips_gazebo)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config urdf worlds
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS scripts/reset_door_test.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
