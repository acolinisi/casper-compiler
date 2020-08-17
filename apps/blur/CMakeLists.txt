include(casper)

casper_add_exec(blur blur.meta
  HALIDE_GENERATORS halide_blur
  NODE_TYPE_IDS 0 1
  SOURCES blur.meta.cpp halide_kern.meta.cpp)
casper_add_c_kern(blur kern kern.c)

# Run the target program (test target)
add_custom_target(blur_run COMMAND blur)
