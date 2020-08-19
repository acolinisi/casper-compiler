cmake_minimum_required(VERSION 3.10)
project(main)

# TensorFlow is not distributed with a FindTensorFlow.cmake, so pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(TensorFlow REQUIRED IMPORTED_TARGET tensorflow)

set(CMAKE_CXX_STANDARD 17)

add_executable(save save.cpp hardware.cpp kernel.cpp kernel_map.cpp model.cpp step.cpp src/Model.cpp src/Tensor.cpp)
add_executable(load load.cpp hardware.cpp kernel.cpp kernel_map.cpp model.cpp step.cpp src/Model.cpp src/Tensor.cpp)
target_link_libraries (save PkgConfig::TensorFlow)
target_link_libraries (load PkgConfig::TensorFlow)
