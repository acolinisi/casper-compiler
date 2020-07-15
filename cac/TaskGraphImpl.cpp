#include "TaskGraphImpl.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace cac {

ValueImpl::ValueImpl(enum ValueType type) : type(type) { }
ValueImpl::~ValueImpl() { }

ScalarImpl::ScalarImpl(enum ScalarImpl::ScalarType type, bool initialized)
	: ValueImpl(ValueImpl::Scalar), initialized(initialized) { }

IntScalarImpl::IntScalarImpl(uint8_t width)
	: ScalarImpl(ScalarImpl::Int, false), width(width), v(0) {
}

IntScalarImpl::IntScalarImpl(uint8_t width, uint64_t v)
	: ScalarImpl(ScalarImpl::Int, true), width(width), v(v) {
}

mlir::Type IntScalarImpl::getType(mlir::OpBuilder &builder) {
	return builder.getIntegerType(width);
}
mlir::LLVM::LLVMType IntScalarImpl::getLLVMType(mlir::OpBuilder &builder,
		mlir::LLVM::LLVMDialect *llvmDialect) {
	return mlir::LLVM::LLVMType::getIntNTy(llvmDialect, width);
}

mlir::Attribute IntScalarImpl::getInitValueAttr(mlir::OpBuilder &builder,
		mlir::LLVM::LLVMDialect *llvmDialect) {
	// APInt is necessary, not sure why the uint64_t overload produces 0 value
	return mlir::IntegerAttr::get(getLLVMType(builder, llvmDialect),
			mlir::APInt(width, v));
}

DatImpl::DatImpl(int rows, int cols, const std::vector<double> &vals)
	: ValueImpl(ValueImpl::Dat), rows(rows), cols(cols), vals(vals) {
}

} // namespace cac
