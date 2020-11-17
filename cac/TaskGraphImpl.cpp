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
mlir::LLVM::LLVMType IntScalarImpl::getLLVMType(
		mlir::LLVM::LLVMDialect *llvmDialect) {
	return mlir::LLVM::LLVMType::getIntNTy(llvmDialect, width);
}

mlir::Attribute IntScalarImpl::getInitValue(mlir::OpBuilder &builder) {
	// APInt is necessary, not sure why the uint64_t overload produces 0 value
	return mlir::IntegerAttr::get(getType(builder), mlir::APInt(width, v));
}

DoubleScalarImpl::DoubleScalarImpl(double v)
	: ScalarImpl(ScalarImpl::Double, true), v(v) {
}
mlir::Type DoubleScalarImpl::getType(mlir::OpBuilder &builder) {
	return builder.getF64Type();
}
mlir::LLVM::LLVMType DoubleScalarImpl::getLLVMType(
		mlir::LLVM::LLVMDialect *llvmDialect) {
	return mlir::LLVM::LLVMType::getDoubleTy(llvmDialect);
}
mlir::Attribute DoubleScalarImpl::getInitValue(mlir::OpBuilder &builder) {
	// APDouble is necessary, not sure why the uint64_t overload produces 0 value
	return mlir::FloatAttr::get(getType(builder), mlir::APFloat(v));
}

DatImpl::DatImpl(int dims, std::vector<int> size)
	: ValueImpl(ValueImpl::Dat), dims(dims), size(size) {
}

mlir::Type DoubleDatImpl::getElementType(mlir::OpBuilder &builder) {
	return builder.getF64Type();
}
mlir::Type FloatDatImpl::getElementType(mlir::OpBuilder &builder) {
	return builder.getF32Type();
}

PyObjImpl::PyObjImpl()
	: ValueImpl(ValueImpl::PyObj) {
}

} // namespace cac
