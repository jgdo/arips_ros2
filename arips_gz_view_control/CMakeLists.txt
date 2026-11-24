cmake_minimum_required(VERSION 3.22)
project(arips_gz_view_control)

# Gazebo GUI plugin: gz-gui's InteractiveViewControl with left / middle mouse buttons swapped
# (left drag = orbit, middle drag = pan). Loaded by the Gazebo GUI as "AripsViewControl" from
# GZ_GUI_PLUGIN_PATH, which the environment hook below extends.

find_package(ament_cmake REQUIRED)
find_package(gz_gui_vendor REQUIRED)
find_package(gz-gui REQUIRED)
find_package(gz_rendering_vendor REQUIRED)
find_package(gz-rendering REQUIRED)
find_package(gz_transport_vendor REQUIRED)
find_package(gz-transport REQUIRED)
find_package(gz_msgs_vendor REQUIRED)
find_package(gz-msgs REQUIRED)
find_package(gz_plugin_vendor REQUIRED)
find_package(gz-plugin REQUIRED COMPONENTS register)
find_package(gz_common_vendor REQUIRED)
find_package(gz-common REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Core Gui Qml Quick)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

add_library(AripsViewControl SHARED
  src/AripsViewControl.cc
  src/AripsViewControl.hh
  src/AripsViewControl.qrc
)
target_compile_features(AripsViewControl PRIVATE cxx_std_17)
target_link_libraries(AripsViewControl PRIVATE
  gz-gui::gz-gui
  gz-rendering::gz-rendering
  gz-transport::gz-transport
  gz-msgs::gz-msgs
  gz-plugin::register
  gz-common::gz-common
  Qt6::Core Qt6::Gui Qt6::Qml Qt6::Quick
)

install(TARGETS AripsViewControl LIBRARY DESTINATION lib/${PROJECT_NAME})

ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/hooks/${PROJECT_NAME}.dsv.in")

ament_package()
