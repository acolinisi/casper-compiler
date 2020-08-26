include(casper)

casper_add_exec(blur blur.meta
  SOURCES blur.meta.cpp halide_kern.meta.cpp
  C_KERNEL_SOURCES kern.c
  BUILD_DIR target)

# Run the target program (test target)
add_custom_target(blur_run COMMAND target/blur DEPENDS target/blur)
