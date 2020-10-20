include(casper)

casper_add_exec(ch ch.meta
  SOURCES ch.meta.cpp
  PLATFORM platform.ini
  TUNED_PARAMS tuned-params.ini
  )

# TODO: workaround: the app has to set Python_SITELIB explicitly.
# The site-packages subdir not appended automatically, even if we import
# 'site' module, and even if wa call site.main() -- double-check the latter
# part, though.
find_package(Python COMPONENTS Interpreter REQUIRED) # for Python_SITELIB

# Run the target program (test target)
#add_custom_target(ch.run COMMAND ${CMAKE_COMMAND} -E env
#  PY_KERNEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}:${Python_SITELIB}" ./ch
#  DEPENDS ch)

# Run the target program (test target)
add_custom_target(ch.run
  ${CMAKE_COMMAND} -E env
    PY_KERNEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}:${Python_SITELIB}"
    mpirun -n 2 ${ch_BUILD_DIR}/ch
  DEPENDS ${ch_BUILD_DIR}/ch)
