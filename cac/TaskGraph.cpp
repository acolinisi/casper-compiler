#include "TaskGraph.h"
#include "DatImpl.h"

namespace cac {

Dat::Dat(int rows, int cols, const std::vector<double> &vals)
: rows(rows), cols(cols), vals(vals), impl(new DatImpl())
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
	dats.push_back(std::move(dat));
	return *dats.back();
}

Task& TaskGraph::createTask(CKernel kern, std::vector<Dat *> dats,
				std::vector<Task*> deps)
{
	return createTask(new CTask(kern.func, dats), dats, deps);
}
Task& TaskGraph::createTask(HalideKernel kern, std::vector<Dat *> dats,
				std::vector<Task*> deps)
{
	return createTask(new HalideTask(kern.func, dats), dats, deps);
}

Task& TaskGraph::createTask(Task *taskp, std::vector<Dat *> dats,
				std::vector<Task*> deps)
{
	std::unique_ptr<Task> task(taskp);
	for (Task* dep : deps) {
		task->deps.push_back(dep);
	}
	tasks.push_back(std::move(task));
	return *tasks.back();
}

} /* namespace cac */
