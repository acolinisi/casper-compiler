add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Support
  )

#get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
#get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh5CombineIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

add_llvm_executable(cac
  app.cpp
  build.cpp
  cac.cpp
  TaskGraph.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/LowerToAffineLoops.cpp
  mlir/LowerToLLVM.cpp
  mlir/ShapeInferencePass.cpp
  mlir/ToyCombine.cpp

  DEPENDS
  ToyCh5ShapeInferenceInterfaceIncGen
  ToyCh5OpsIncGen
  ToyCh5CombineIncGen
)
include_directories(include/)
llvm_update_compile_flags(cac)
target_link_libraries(cac
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRAffineToStandard
  MLIRAnalysis
  MLIRCallInterfaces
  MLIRExecutionEngine
  MLIRIR
  MLIRLLVMIR
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRSCFToStandard
  MLIRStandardToLLVM
  MLIRSupport
  MLIRTargetLLVMIR
  MLIRTransforms
  )

mlir_check_link_libraries(cac)
