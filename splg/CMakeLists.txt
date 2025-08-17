cmake_minimum_required(VERSION 3.22.1)
project(splg VERSION 1.0.0 LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-unknown-pragmas -Xcompiler -Wno-cpp")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(CUDAToolkit REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui cudaarithm cudaimgproc)

# Find TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

set(TENSORRT_LIBRARIES
  ${TENSORRT_LIBRARY_INFER}
  ${TENSORRT_LIBRARY_ONNXPARSER}
  nvToolsExt
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY_INFER)
  message(FATAL_ERROR "TensorRT not found. Please specify TENSORRT_ROOT.")
endif()

add_library(splg_component SHARED
    src/splg_trt_component.cpp
    src/type_conv_helper.cu
)

set_property(TARGET splg_component PROPERTY CXX_STANDARD 17)
set_property(TARGET splg_component PROPERTY CUDA_STANDARD 17)

ament_target_dependencies(splg_component
  rclcpp
  rclcpp_components
  sensor_msgs
  cv_bridge
  message_filters
)

target_include_directories(splg_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  ${CUDAToolkit_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(splg_component
  ${OpenCV_LIBRARIES}
  ${TENSORRT_LIBRARIES}
  ${CUDAToolkit_LIBRARIES}
  CUDA::nvToolsExt
)

rclcpp_components_register_nodes(splg_component "uosm::SuperPointLightGlueTrt")

add_executable(splg_node
  src/splg_trt_node.cpp
)

ament_target_dependencies(splg_node rclcpp)
target_link_libraries(splg_node splg_component)

install(TARGETS splg_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(DIRECTORY include/
  DESTINATION include
)

install(TARGETS splg_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch weights
  DESTINATION share/${PROJECT_NAME}/
)

ament_export_dependencies(
  rclcpp
  rclcpp_components
  sensor_msgs
  cv_bridge
  message_filters
)
ament_export_include_directories(include)
ament_export_libraries(splg_component)

ament_package()