set(LLVM_LINK_COMPONENTS
  Support
  )

#get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
#get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(cac
  cac.cpp
  parser/AST.cpp
  )
include_directories(include/)
llvm_update_compile_flags(cac)
target_link_libraries(cac
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRSupport
  )

mlir_check_link_libraries(cac)
