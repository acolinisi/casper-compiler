set(lib casper_runtime)
set(lib_prof casper_runtime_prof) # for profiling harness

set(COMMON_SOURCES
  monitor.cpp
  platform.cpp
  python.cpp
)

add_library(${lib} SHARED ${COMMON_SOURCES})

add_library(${lib_prof} SHARED
  ${COMMON_SOURCES}
  profile.cpp
)

find_package(Halide REQUIRED)

# TODO: not all apps need python, make optional
# Settings common to both libs
foreach (rtlib ${lib} ${lib_prof})
  find_package(Python COMPONENTS Development REQUIRED)
  target_include_directories(${rtlib} PRIVATE ${Python_INCLUDE_DIRS})
  target_link_directories(${rtlib} PRIVATE ${Python_LIBRARY_DIRS})
  target_link_libraries(${rtlib} ${Python_LIBRARIES} Halide::Tools)
endforeach()

target_link_libraries(${lib_prof} Halide::Tools)
