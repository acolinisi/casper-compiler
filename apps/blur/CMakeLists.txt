include(casper)

casper_add_exec(blur blur.meta blur.meta.cpp)
casper_add_c_kern(blur kern kern.c)
casper_add_halide_meta(blur GENERATORS halide_blur
  SOURCES halide_kern.meta.cpp)

# Run the target program (test target)
add_custom_target(blur_run COMMAND blur)
