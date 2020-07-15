#ifndef CAC_TASK_GRAPH_IMPL_H
#define CAC_TASK_GRAPH_IMPL_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace cac {

class ValueImpl {
public:
	mlir::Value ref;
};

class ScalarImpl : public ValueImpl {
public:
	virtual mlir::Type getType(mlir::OpBuilder &builder) = 0;
	virtual mlir::LLVM::LLVMType getLLVMType(mlir::OpBuilder &builder,
	    mlir::LLVM::LLVMDialect *llvmDialect) = 0;
	virtual mlir::Attribute getInitValueAttr(mlir::OpBuilder &builder,
	    mlir::LLVM::LLVMDialect *llvmDialect) = 0;
};

class IntScalarImpl : public ScalarImpl {
public:
	IntScalarImpl(uint8_t width);
	IntScalarImpl(uint8_t width, uint64_t v);

	virtual mlir::Type getType(mlir::OpBuilder &builder);
	virtual mlir::LLVM::LLVMType getLLVMType(mlir::OpBuilder &builder,
	    mlir::LLVM::LLVMDialect *llvmDialect);
	virtual mlir::Attribute getInitValueAttr(mlir::OpBuilder &builder,
	    mlir::LLVM::LLVMDialect *llvmDialect);
public:
  // Owner class tracks whether value initialized
  const uint64_t v; // type large enough for max width
  const uint8_t width;
};

class DatImpl : public ValueImpl { };

} // nameaspace cac

#endif // CAC_TASK_GRAPH_IMPL_H
