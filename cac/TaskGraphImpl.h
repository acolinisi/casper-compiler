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
  virtual mlir::Value load(mlir::OpBuilder &builder);
public:
  enum ValueType type;
  mlir::Type stdTy;
  mlir::Value ptr;
  mlir::Value ref;
};

class ScalarImpl : public ValueImpl {
public:
  enum ScalarType {
    Int,
    Double,
    Ptr,
  };
public:
  ScalarImpl(enum ScalarType type, bool initialized);
  virtual mlir::Type getType(mlir::OpBuilder &builder) = 0;
  virtual mlir::LLVM::LLVMType getLLVMType(
      mlir::LLVM::LLVMDialect *llvmDialect) = 0;
  virtual mlir::Attribute getInitValue(mlir::OpBuilder &builder) = 0;
  virtual bool isPointer();
  virtual mlir::Value getPtr();
  virtual mlir::Value load(mlir::OpBuilder &builder);
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

class DoubleScalarImpl : public ScalarImpl {
public:
  DoubleScalarImpl(double v);

  virtual mlir::Type getType(mlir::OpBuilder &builder);
  virtual mlir::LLVM::LLVMType getLLVMType(
      mlir::LLVM::LLVMDialect *llvmDialect);
  virtual mlir::Attribute getInitValue(mlir::OpBuilder &builder);
public:
  const double v;
};

class PtrScalarImpl : public ScalarImpl {
public:
  PtrScalarImpl(ScalarImpl *dest);

  virtual mlir::Type getType(mlir::OpBuilder &builder);
  virtual mlir::LLVM::LLVMType getLLVMType(
      mlir::LLVM::LLVMDialect *llvmDialect);
  virtual mlir::Attribute getInitValue(mlir::OpBuilder &builder);
  virtual bool isPointer();
  virtual mlir::Value getPtr();
  virtual mlir::Value load(mlir::OpBuilder &builder);
public:
  ScalarImpl *dest;
};

class DatImpl : public ValueImpl {
protected:
  DatImpl(int dims, const std::vector<int> size);

public:
  virtual mlir::Type getElementType(mlir::OpBuilder &builder) = 0;
public:
  const int dims;
  std::vector<int> size;

};

class DoubleDatImpl : public DatImpl {
public:
  DoubleDatImpl(int dims, const std::vector<int> size,
      const std::vector<double> &vals = {})
    : DatImpl(dims, size), vals(vals) {}

  virtual mlir::Type getElementType(mlir::OpBuilder &builder);
public:
  std::vector<double> vals;
};

class FloatDatImpl : public DatImpl {
public:
  FloatDatImpl(int dims, const std::vector<int> size,
      const std::vector<float> &vals = {})
    : DatImpl(dims, size), vals(vals) {}

  virtual mlir::Type getElementType(mlir::OpBuilder &builder);
public:
  std::vector<float> vals;
};

class IntDatImpl : public DatImpl {
public:
  IntDatImpl(int width, int dims, const std::vector<int> size,
      const std::vector<double> &vals = {})
    : DatImpl(dims, size), vals(vals), width(width) {}

  virtual mlir::Type getElementType(mlir::OpBuilder &builder);
public:
  std::vector<double> vals;
  int width;
};

class PyObjImpl : public ValueImpl {
public:
  PyObjImpl();
};

class PyTaskImpl {
public:
    mlir::Value generatorContext;
};

class HalideTaskImpl {
public:
	std::vector<std::string> params;
	std::vector<std::string> inputProps;
};

} // nameaspace cac

#endif // CAC_TASK_GRAPH_IMPL_H
