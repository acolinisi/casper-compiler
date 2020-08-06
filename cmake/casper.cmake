set(CASPER_COMPILER_LIB cac)

# Create a Casper executable (application)
# Arguments: app_target SOURCES source_file... HALIDE_GENERATORS gen1 ...
function(casper_add_exec target meta_prog)
	cmake_parse_arguments(FARG
		""
		""
		"SOURCES;HALIDE_GENERATORS" ${ARGN})

	find_package(Threads)

	# TODO: FindCasper.cmake (figure out if functions go into Find*.cmake),
	# then move these out of the function
	find_program(LLC llc REQUIRED DOC "LLVM IR compiler")
	include_directories(${CAC_INCLUDE_DIRS})

	add_executable(${meta_prog} ${FARG_SOURCES})
	target_link_libraries(${meta_prog} LINK_PUBLIC ${CASPER_COMPILER_LIB})

	foreach(gen ${FARG_HALIDE_GENERATORS})
		set(halide_lib lib${gen}.a)
		list(APPEND halide_libs ${halide_lib})
	endforeach()

	list(APPEND halide_libs libhalide_runtime.a)

	foreach(halide_lib ${halide_libs})
		list(APPEND halide_libs_paths
			${CMAKE_CURRENT_BINARY_DIR}/${halide_lib})
	endforeach()

	# Run the meta-program
	add_custom_command(OUTPUT ${target}.ll ${halide_libs}
	  COMMAND ${meta_prog} 2> ${target}.ll
	  DEPENDS ${meta_prog})

	# Compile the target harness
	add_custom_command(OUTPUT ${target}.o
	  COMMAND ${LLC} -filetype=obj -o ${target}.o ${target}.ll
	  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${target}.ll)

	# Link the binary
	add_executable(${target} ${CMAKE_CURRENT_BINARY_DIR}/${target}.o)
	target_link_libraries(${target} ${halide_libs_paths}
		casper_runtime
		Threads::Threads ${CMAKE_DL_LIBS})
	set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

# Add to app kernels written in C or C++ with C ABI
# Arguments: app_target source_file...
function(casper_add_c_kern app)
	set(lib ${app}_kern_c)
	add_library(${lib} ${ARGN})
	target_link_libraries(${app} ${lib})
endfunction()
