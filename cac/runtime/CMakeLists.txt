set(lib casper_runtime)
add_library(${lib} SHARED platform.cpp python.cpp)

# TODO: not all apps need python, make optional
find_package(Python COMPONENTS Development REQUIRED)
target_include_directories(${lib} PRIVATE ${Python_INCLUDE_DIRS})
target_link_directories(${lib} PRIVATE ${Python_LIBRARY_DIRS})
target_link_libraries(${lib} ${Python_LIBRARIES})
