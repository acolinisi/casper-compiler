include(casper)

casper_add_exec(cavity cavity.meta
  SOURCES cavity.meta.cpp)

# Run the target program (test target)
add_custom_target(cavity_run COMMAND ${CMAKE_COMMAND} -E env
  PY_KERNEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}" ./cavity
  DEPENDS cavity)
