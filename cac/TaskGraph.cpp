#include "TaskGraph.h"
#include "TaskGraphImpl.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace cac {

Value::Value(enum ValueType type, ValueImpl *impl) : type(type), impl(impl)
{}

Scalar::Scalar(enum ScalarType type, bool initialized, ScalarImpl *impl)
	: Value(Value::Scalar, impl), type(type), initialized(initialized),
	impl(impl)
{}

IntScalar::IntScalar(uint8_t width)
	: Scalar(Scalar::Int, false, new IntScalarImpl(width))
{}
IntScalar::IntScalar(uint8_t width, uint64_t v)
	: Scalar(Scalar::Int, true, new IntScalarImpl(width, v))
{}

Dat::Dat(int rows, int cols, const std::vector<double> &vals)
: Value(Value::Dat, new DatImpl()), rows(rows), cols(cols), vals(vals)
{ }

Dat::~Dat()
{
	delete impl;
}

Dat& TaskGraph::createDat(int n, int m) {
	return createDat(n, m, {});
}
Dat& TaskGraph::createDat(int n, int m, const std::vector<double> &vals)
{
	std::unique_ptr<Dat> dat(new Dat(n, m, vals));
	Dat& ref = *dat;
	values.push_back(std::move(dat));
	//return *values.back();
	return ref;
}
IntScalar& TaskGraph::createIntScalar(uint8_t width) {
	std::unique_ptr<IntScalar> scalar(new IntScalar(width));
	return createIntScalar(std::move(scalar));
}
IntScalar& TaskGraph::createIntScalar(uint8_t width, uint64_t v) {
	std::unique_ptr<IntScalar> scalar(new IntScalar(width, v));
	return createIntScalar(std::move(scalar));
}
IntScalar& TaskGraph::createIntScalar(std::unique_ptr<IntScalar> scalar) {
	IntScalar& ref = *scalar;
	values.push_back(std::move(scalar));
	return ref;
}

Task& TaskGraph::createTask(CKernel kern, std::vector<Value *> args,
				std::vector<Task*> deps)
{
	std::unique_ptr<Task> task(new CTask(kern.func, args));
	return createTask(std::move(task), deps);
}
Task& TaskGraph::createTask(HalideKernel kern, std::vector<Value *> args,
				std::vector<Task*> deps)
{
	std::unique_ptr<Task> task(new HalideTask(kern.func, args));
	return createTask(std::move(task), deps);
}

Task& TaskGraph::createTask(std::unique_ptr<Task> task, std::vector<Task*> deps)
{
	for (Task* dep : deps) {
		task->deps.push_back(dep);
	}
	tasks.push_back(std::move(task));
	return *tasks.back();
}

IntScalarImpl::IntScalarImpl(uint8_t width)
	: width(width), v(0) {
}

IntScalarImpl::IntScalarImpl(uint8_t width, uint64_t v)
	: width(width), v(v) {
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

} /* namespace cac */
