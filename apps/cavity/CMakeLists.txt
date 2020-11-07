include(casper)

casper_add_exec(cavity cavity.meta
  SOURCES cavity.meta.cpp)

# TODO: workaround: the app has to set Python_SITELIB explicitly.
# The site-packages subdir not appended automatically, even if we import
# 'site' module, and even if wa call site.main() -- double-check the latter
# part, though.
find_package(Python COMPONENTS Interpreter REQUIRED) # for Python_SITELIB

# Run the target program (test target)
add_custom_target(cavity_run COMMAND ${CMAKE_COMMAND} -E env
  EXTRA_PYTHONPATH="${CMAKE_CURRENT_SOURCE_DIR}:${Python_SITELIB}" ./cavity
  DEPENDS cavity)
