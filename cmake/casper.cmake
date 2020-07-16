set(CASPER_COMPILER_LIB cac)

# Create a Casper executable (application)
# Arguments: app_target source_file...
function(casper_add_exec target meta_prog)
	#cmake_parse_arguments(FARG
	#	""
	#	""
	#	SOURCES ${ARGN})

	# TODO: FindCasper.cmake (figure out if functions go into Find*.cmake),
	# then move these out of the function
	find_program(LLC llc REQUIRED DOC "LLVM IR compiler")
	include_directories(${CAC_INCLUDE_DIRS})

	add_executable(${meta_prog} ${ARGN})
	target_link_libraries(${meta_prog} LINK_PUBLIC ${CASPER_COMPILER_LIB})

	# Run the meta-program
	add_custom_command(OUTPUT ${target}.ll
	  COMMAND ${meta_prog} 2> ${target}.ll
	  DEPENDS ${meta_prog})

	# Compile and link the target binary
	add_custom_command(OUTPUT ${target}.o
	  COMMAND ${LLC} -filetype=obj -o ${target}.o ${target}.ll
	  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${target}.ll)

	add_executable(${target} ${CMAKE_CURRENT_BINARY_DIR}/${target}.o)
	set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

# Add to app kernels written in C or C++ with C ABI
# Arguments: app_target source_file...
function(casper_add_c_kern app)
	set(lib ${app}_kern_c)
	add_library(${lib} ${ARGN})
	target_link_libraries(${app} ${lib})
endfunction()

# Add to app kernels written as Halide pipelines
# Arguments: app_target GENERATORS gen... SOURCES source_file... PARAMS p,...
function(casper_add_halide_meta app)
	cmake_parse_arguments(FARG
		""
		""
		"GENERATORS;SOURCES;PARAMS" ${ARGN})

	find_program(LLC llc REQUIRED DOC "LLVM IR compiler")

	# TODO: in FindCasper.cmake, take option HALIDE
	find_package(Halide REQUIRED)
	find_package(OpenMP)

	# Target to build the generator
	add_executable(${app}.halide_meta ${FARG_SOURCES})
	target_link_libraries(${app}.halide_meta PRIVATE Halide::Generator)

	# Target to run the generator
	foreach(gen ${FARG_GENERATORS})
		# TODO: name library ${app}_kern_halide_X
		add_halide_library(${gen} FROM ${app}.halide_meta
						   STMT ${gen}_STMT
						   LLVM_ASSEMBLY ${gen}_LLVM_ASSEMBLY
						   PARAMS ${FARG_PARAMS})
		target_link_libraries(${app} ${gen})
	endforeach()
endfunction()
