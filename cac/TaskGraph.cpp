#include "TaskGraph.h"
#include "TaskGraphImpl.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace cac {

Value::Value(ValueImpl *impl) : impl(impl) {}
Value::~Value() {
	delete impl;
}
ValueImpl *Value::getImpl() {
	return impl;
}

Scalar::Scalar(ScalarImpl *impl) : Value(impl)
{}
ScalarImpl *Scalar::getImpl() {
	return static_cast<ScalarImpl *>(impl);
}

IntScalar::IntScalar(uint8_t width)
	: Scalar(new IntScalarImpl(width))
{}
IntScalar::IntScalar(uint8_t width, uint64_t v)
	: Scalar(new IntScalarImpl(width, v))
{}

Dat::Dat(DatImpl *impl) : Value(impl) { }
DatImpl *Dat::getImpl() {
	return static_cast<DatImpl *>(impl);
}

Dat& TaskGraph::createDat(int n, int m) {
	return createDat(n, m, {});
}
Dat& TaskGraph::createDat(int n, int m, const std::vector<double> &vals)
{
	std::unique_ptr<Dat> dat(new Dat(new DatImpl(n, m, vals)));
	Dat& ref = *dat;
	values.push_back(std::move(dat));
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

} /* namespace cac */
