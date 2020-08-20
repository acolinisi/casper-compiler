# TensorFlow is not distributed with a FindTensorFlow.cmake, so pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(TensorFlow REQUIRED IMPORTED_TARGET tensorflow)

add_library(knowbase SHARED
  hardware.cpp
  kernel.cpp
  kernel_map.cpp
  load.cpp
  model.cpp
  save.cpp
  step.cpp
  Model.cpp
  Tensor.cpp
)

target_link_libraries(knowbase PkgConfig::TensorFlow)
target_include_directories(knowbase PUBLIC include)
#include_directories(include)
