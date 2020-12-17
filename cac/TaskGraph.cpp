#include "TaskGraph.h"
#include "TaskGraphImpl.h"

namespace cac {

HalideTask::HalideTask(const std::string &func, std::vector<Value *> args)
	: Task(Task::Halide, func, args), impl(new HalideTaskImpl()) {}

PyTask::PyTask(PyTask::Type type, const std::string &module,
	const std::string &func, std::vector<Value *> args)
	: Task(Task::Python, func, args), type(type),
	module(module), impl(new PyTaskImpl()) {}

PyTask::PyTask(const std::string &module, const std::string &func,
	std::vector<Value *> args)
	: PyTask(PyTask::Function, module, func, args) {}

PyGenedTask::PyGenedTask(const std::string &module,
		const std::string &kernel,
	std::vector<Value *> args)
	: PyTask(PyTask::Generated, module, "", args), kernel(kernel) {}

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
DoubleScalar::DoubleScalar(double v)
	: Scalar(new DoubleScalarImpl(v))
{}

Dat::Dat(DatImpl *impl) : Value(impl) { }
DatImpl *Dat::getImpl() {
	return static_cast<DatImpl *>(impl);
}

PyObj::PyObj(PyObjImpl *impl) : Value(impl) { }
PyObjImpl *PyObj::getImpl() {
	return static_cast<PyObjImpl *>(impl);
}

TaskGraph::TaskGraph(const std::string &name) :
	name(name), datPrintEnabled(false) { }

Dat& TaskGraph::createDat(int n, int m) {
	return createDat(n, m, {});
}
Dat& TaskGraph::createDat(int n, int m, const std::vector<double> &vals)
{
	// TODO: vals
	return createDoubleDat(2, {n, m});
}

Dat& TaskGraph::createDoubleDat(int dims, const std::vector<int> size)
{
	std::unique_ptr<Dat> dat(new DoubleDat(
		new DoubleDatImpl(dims, size)));
	Dat& ref = *dat;
	values.push_back(std::move(dat));
	return ref;
}
Dat& TaskGraph::createFloatDat(int dims, const std::vector<int> size)
{
	std::unique_ptr<Dat> dat(new FloatDat(
		new FloatDatImpl(dims, size)));
	Dat& ref = *dat;
	values.push_back(std::move(dat));
	return ref;
}
Dat& TaskGraph::createIntDat(int width, int dims,
		const std::vector<int> size)
{
	std::unique_ptr<Dat> dat(new IntDat(
		new IntDatImpl(width, dims, size)));
	Dat& ref = *dat;
	values.push_back(std::move(dat));
	return ref;
}

PyObj& TaskGraph::createPyObj()
{
	std::unique_ptr<PyObj> pyObj(new PyObj(new PyObjImpl()));
	PyObj& ref = *pyObj;
	values.push_back(std::move(pyObj));
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
DoubleScalar& TaskGraph::createDoubleScalar(double v) {
	std::unique_ptr<DoubleScalar> scalar(new DoubleScalar(v));
	DoubleScalar& ref = *scalar;
	values.push_back(std::move(scalar));
	return ref;
}

Task& TaskGraph::createTask(CKernel kern, std::vector<Value *> args,
				std::vector<Task*> deps)
{
	std::unique_ptr<Task> task(new CTask(kern.func, args));
	return createTask(std::move(task), deps);
}
Task& TaskGraph::createTask(PyKernel kern, std::vector<Value *> args,
				std::vector<Task*> deps)
{
	std::unique_ptr<Task> task(new PyTask(kern.module, kern.func, args));
	return createTask(std::move(task), deps);
}
Task& TaskGraph::createTask(FEMAKernel kern, std::vector<Value *> args,
				std::vector<Task*> deps)
{
	std::unique_ptr<Task> task(new PyGenedTask(kern.module, kern.func, args));
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

void TaskGraph::registerPyGenerator(const std::string &mod,
		const std::string &func) {
	std::unique_ptr<PyGenerator> gen(new PyGenerator{mod, func});
	pyGenerators.push_back(std::move(gen));
}

void TaskGraph::setDatPrint(bool enable) {
	datPrintEnabled = enable;
}

} /* namespace cac */
