cmake_minimum_required(VERSION 3.22.1) 
project(oakd_ros2_minimal VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_compile_options(-g)

if(POLICY CMP0057)
    cmake_policy(SET CMP0057 NEW)
endif()

set(_opencv_version 4)
find_package(OpenCV 4 QUIET COMPONENTS imgproc highgui)
if(NOT OpenCV_FOUND)
  set(_opencv_version 3)
  find_package(OpenCV 3 REQUIRED COMPONENTS imgproc highgui)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

set(DEPENDENCIES
  camera_info_manager
  cv_bridge
  depthai_ros_msgs
  depthai_bridge
  depthai
  rclcpp
  rclcpp_components
  sensor_msgs
  stereo_msgs
  std_msgs
  vision_msgs
  tf2
  tf2_geometry_msgs
)

set(OAK_CAMERA_INC
  ${CMAKE_CURRENT_SOURCE_DIR}/include/oak_camera_component.hpp
)

set(OAK_CAMERA_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/oak_camera_component.cpp
)

add_library(oak_camera_component SHARED
    ${OAK_CAMERA_INC}
    ${OAK_CAMERA_SRC}
)

ament_target_dependencies(oak_camera_component ${DEPENDENCIES})

rclcpp_components_register_nodes(oak_camera_component "uosm::depthai::OakCamera")

install(TARGETS oak_camera_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION include/${PROJECT_NAME}/
)

install(DIRECTORY
  ${CMAKE_CURRENT_SOURCE_DIR}/launch/ 
  DESTINATION share/${PROJECT_NAME}/launch
)

install(PROGRAMS
  scripts/save_images.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(
    oak_camera_component,
)
ament_export_dependencies(${DEPENDENCIES})
ament_package()
