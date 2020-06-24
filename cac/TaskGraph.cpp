#include "TaskGraph.h"

namespace cac {

Dat& TaskGraph::createDat(int n, int m, const std::vector<double> &vals)
{
	std::unique_ptr<Dat> dat(new Dat(n, m, vals));
	dats.push_back(std::move(dat));
	return *dats.back();
}

Task& TaskGraph::createTask(const std::string &func, Dat &dat)
{
	std::vector<Task *> deps;
	return this->createTask(func, dat, deps);
}

Task& TaskGraph::createTask(const std::string &func, Dat &dat,
				std::vector<Task*> deps)
{
	std::unique_ptr<Task> task(new Task(func, dat));
	for (Task* dep : deps) {
		//dep->dependees.push_back(task.get());
		task->deps.push_back(dep);
	}
	tasks.push_back(std::move(task));
	return *tasks.back();
}

} /* namespace cac */
