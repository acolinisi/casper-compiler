add_executable(matrix.meta
  matrix.meta.cpp
)

include_directories(${CAC_INCLUDE_DIRS})

target_link_libraries(matrix.meta LINK_PUBLIC cac)

add_library(kern kern.c)

add_custom_command(OUTPUT matrix.ll
  COMMAND matrix.meta
  COMMAND matrix.meta 2> matrix.ll
  DEPENDS matrix.meta)

find_program(LLC llc REQUIRED DOC "LLVM IR compiler")

add_custom_command(OUTPUT matrix.o
  COMMAND ${LLC} -filetype=obj -o matrix.o matrix.ll
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/matrix.ll)

add_executable(matrix
  ${CMAKE_CURRENT_BINARY_DIR}/matrix.o
)
set_target_properties(matrix PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(matrix kern)

add_custom_target(matrix_run COMMAND matrix)
