#ifndef CAC_TASK_GRAPH_IMPL_H
#define CAC_TASK_GRAPH_IMPL_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace cac {

class ValueImpl {
public:
  enum ValueType {
    Scalar,
    Dat,
    PyObj,
  };
public:
  ValueImpl(enum ValueType type);
  virtual ~ValueImpl();
public:
  enum ValueType type;
  mlir::Value ref;
};

class ScalarImpl : public ValueImpl {
public:
  enum ScalarType {
    Int,
    Float,
  };
public:
  ScalarImpl(enum ScalarType type, bool initialized);
  virtual mlir::Type getType(mlir::OpBuilder &builder) = 0;
  virtual mlir::LLVM::LLVMType getLLVMType(
      mlir::LLVM::LLVMDialect *llvmDialect) = 0;
  virtual mlir::Attribute getInitValue(mlir::OpBuilder &builder) = 0;
public:
  enum ScalarType type;
  const bool initialized;
};

class IntScalarImpl : public ScalarImpl {
public:
	IntScalarImpl(uint8_t width);
	IntScalarImpl(uint8_t width, uint64_t v);

	virtual mlir::Type getType(mlir::OpBuilder &builder);
	virtual mlir::LLVM::LLVMType getLLVMType(
	    mlir::LLVM::LLVMDialect *llvmDialect);
	virtual mlir::Attribute getInitValue(mlir::OpBuilder &builder);
public:
  const uint8_t width;
  const uint64_t v; // type large enough for max width
};

class DatImpl : public ValueImpl {
public:
  DatImpl(int rows, int cols, const std::vector<double> &vals);
public:
  const int rows, cols;
  std::vector<double> vals;
};

class PyObjImpl : public ValueImpl {
public:
  PyObjImpl();
};

class HalideTaskImpl {
public:
	std::vector<std::string> params;
	std::vector<std::string> inputProps;
};

} // nameaspace cac

#endif // CAC_TASK_GRAPH_IMPL_H
