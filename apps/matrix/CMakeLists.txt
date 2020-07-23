include(casper)

casper_add_exec(matrix matrix.meta matrix.meta.cpp)
casper_add_c_kern(matrix kern kern.c)
casper_add_halide_meta(matrix GENERATORS halide_bright
  SOURCES halide_kern.meta.cpp)

# Run the target program (test target)
add_custom_target(matrix_run COMMAND matrix)
