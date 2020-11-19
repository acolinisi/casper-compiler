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
# TODO: burn the EXTRA_PYTHONPATH into the metaprogram somehow
add_custom_target(ch.run
  ${CMAKE_COMMAND} -E env
    EXTRA_PYTHONPATH="${CMAKE_CURRENT_SOURCE_DIR}:${CAC_PYAPI_DIR}:${Python_SITELIB}"
    mpirun -n $$RANKS ${ch_BUILD_DIR}/ch
    #${ch_BUILD_DIR}/ch
  DEPENDS ${ch_BUILD_DIR}/ch)
