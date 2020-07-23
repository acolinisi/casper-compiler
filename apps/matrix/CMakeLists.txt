include(casper)

casper_add_exec(matrix matrix.meta
  HALIDE_GENERATORS halide_bright
  SOURCES matrix.meta.cpp halide_kern.meta.cpp
)
casper_add_c_kern(matrix kern kern.c)

# Run the target program (test target)
add_custom_target(matrix_run COMMAND matrix)
