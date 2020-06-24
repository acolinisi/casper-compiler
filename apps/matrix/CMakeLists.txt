add_executable(matrix.meta
  matrix.meta.cpp
)

include_directories(${CAC_INCLUDE_DIRS})

target_link_libraries(matrix.meta LINK_PUBLIC cac)
