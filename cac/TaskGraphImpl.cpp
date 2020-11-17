#include "TaskGraphImpl.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace cac {

ValueImpl::ValueImpl(enum ValueType type) : type(type) { }
ValueImpl::~ValueImpl() { }

mlir::Value ValueImpl::load(mlir::OpBuilder &builder) {
	return ref;
}

ScalarImpl::ScalarImpl(enum ScalarImpl::ScalarType type, bool initialized)
	: ValueImpl(ValueImpl::Scalar), initialized(initialized) { }

bool ScalarImpl::isPointer() {
	return false;
}
mlir::Value ScalarImpl::getPtr() {
	return ptr;
}
mlir::Value ScalarImpl::load(mlir::OpBuilder &builder) {
	auto loc = builder.getUnknownLoc(); // TODO
	auto loaded = builder.create<mlir::LLVM::LoadOp>(loc, ptr);
	return builder.create<mlir::LLVM::DialectCastOp>(loc, stdTy, loaded);
}

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

PtrScalarImpl::PtrScalarImpl(ScalarImpl *dest)
	: ScalarImpl(ScalarImpl::Ptr, true), dest(dest) {
}
mlir::Type PtrScalarImpl::getType(mlir::OpBuilder &builder) {
	return dest->getType(builder);
}
mlir::LLVM::LLVMType PtrScalarImpl::getLLVMType(
		mlir::LLVM::LLVMDialect *llvmDialect) {
	return dest->getLLVMType(llvmDialect);
}
mlir::Attribute PtrScalarImpl::getInitValue(mlir::OpBuilder &builder) {
	return dest->getInitValue(builder);
}
bool PtrScalarImpl::isPointer() {
	return true;
}
mlir::Value PtrScalarImpl::getPtr() {
	return dest->getPtr();
}
mlir::Value PtrScalarImpl::load(mlir::OpBuilder &builder) {
	//return dest->load(builder);
	return dest->getPtr();
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
mlir::Type IntDatImpl::getElementType(mlir::OpBuilder &builder) {
	return builder.getIntegerType(width);
}

PyObjImpl::PyObjImpl()
	: ValueImpl(ValueImpl::PyObj) {
}

} // namespace cac
