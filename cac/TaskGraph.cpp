#include "TaskGraph.h"
#include "TaskGraphImpl.h"

namespace cac {

HalideTask::HalideTask(const std::string &func, std::vector<Value *> args)
	: Task(Task::Halide, func, args), impl(new HalideTaskImpl()) {}

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
	std::unique_ptr<Dat> dat(new Dat(new DatImpl(n, m, vals)));
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

void TaskGraph::setDatPrint(bool enable) {
	datPrintEnabled = enable;
}

} /* namespace cac */
