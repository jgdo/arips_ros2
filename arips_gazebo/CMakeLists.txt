cmake_minimum_required(VERSION 3.20)
project(arips_gazebo)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config urdf worlds assets
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
