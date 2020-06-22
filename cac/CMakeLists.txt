add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Support
  )

#get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
#get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(cac
  cac.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp

  DEPENDS
  ToyCh2OpsIncGen
  )
include_directories(include/)
llvm_update_compile_flags(cac)
target_link_libraries(cac
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRAnalysis
  MLIRIR
  MLIRSupport
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms
  )

mlir_check_link_libraries(cac)
