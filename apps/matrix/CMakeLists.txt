# TODO: Move up? (but not all apps need Halide)
find_package(Halide REQUIRED)
find_package(OpenMP)

include_directories(${CAC_INCLUDE_DIRS})

add_executable(matrix.meta
  matrix.meta.cpp
)
target_link_libraries(matrix.meta LINK_PUBLIC cac)

# Kernels written in plain C
add_library(kern kern.c)

# Kernels written as Halide pipelines (build generator, run generator)

add_executable(halide_kern.meta halide_kern.meta.cpp)
target_link_libraries(halide_kern.meta PRIVATE Halide::Generator)
add_halide_library(halide_bright FROM halide_kern.meta
                   STMT halide_bright_STMT
                   LLVM_ASSEMBLY halide_bright_LLVM_ASSEMBLY)

# Run the meta-program

# Note: run twice, the first time just to dump the compiled output
add_custom_command(OUTPUT matrix.ll
  COMMAND matrix.meta
  COMMAND matrix.meta 2> matrix.ll
  DEPENDS matrix.meta)

# Compile and link the target binary

find_program(LLC llc REQUIRED DOC "LLVM IR compiler")

add_custom_command(OUTPUT matrix.o
  COMMAND ${LLC} -filetype=obj -o matrix.o matrix.ll
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/matrix.ll)

add_executable(matrix
  ${CMAKE_CURRENT_BINARY_DIR}/matrix.o
)
set_target_properties(matrix PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(matrix kern halide_bright)

# Run the target program (test target)

add_custom_target(matrix_run COMMAND matrix)
