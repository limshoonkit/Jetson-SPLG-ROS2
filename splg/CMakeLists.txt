cmake_minimum_required(VERSION 3.22.1)
project(splg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

# Find CUDA
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

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

# Create executable
add_executable(splg_node
  src/splg_node.cpp
  src/type_conv_helper.cu
)

target_compile_features(splg_node PUBLIC c_std_99 cxx_std_17)
set_property(TARGET splg_node PROPERTY CUDA_STANDARD 17)
set_property(TARGET splg_node PROPERTY CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-unknown-pragmas -Xcompiler -Wno-cpp")

# Include directories
target_include_directories(splg_node PRIVATE
  ${CUDAToolkit_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(splg_node
  CUDA::cudart
  CUDA::cuda_driver
  nvToolsExt
  ${TENSORRT_LIBRARY_INFER}
  ${TENSORRT_LIBRARY_ONNXPARSER}
  ${OpenCV_LIBRARIES}
)

ament_target_dependencies(splg_node
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
  message_filters
)

# Install
install(TARGETS splg_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch weights
  DESTINATION share/${PROJECT_NAME})

ament_package()