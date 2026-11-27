cmake_minimum_required(VERSION 3.20)
project(arips_description)

find_package(ament_cmake REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(ONSHAPE_RAW_URDF ${CMAKE_CURRENT_SOURCE_DIR}/urdf/arips_description_onshape_raw.urdf)
set(ONSHAPE_XACRO ${CMAKE_CURRENT_BINARY_DIR}/urdf/arips_onshape.urdf.xacro)
set(ONSHAPE_GENERATOR ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_onshape_xacro.py)

add_custom_command(
  OUTPUT ${ONSHAPE_XACRO}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/urdf
  COMMAND ${Python3_EXECUTABLE} ${ONSHAPE_GENERATOR} ${ONSHAPE_RAW_URDF} ${ONSHAPE_XACRO}
  DEPENDS ${ONSHAPE_RAW_URDF} ${ONSHAPE_GENERATOR}
  COMMENT "Generating arips_onshape.urdf.xacro from the Onshape export"
)
add_custom_target(onshape_xacro ALL DEPENDS ${ONSHAPE_XACRO})

install(DIRECTORY urdf launch rviz meshes
  DESTINATION share/${PROJECT_NAME}
)
install(FILES ${ONSHAPE_XACRO}
  DESTINATION share/${PROJECT_NAME}/urdf
)

ament_package()
