#include "TaskGraph.h"

namespace cac {

Dat::Dat(int rows, int cols, const std::vector<double> &vals)
	: rows(rows), cols(cols), vals(vals)
{
}

Task::Task(const std::string &func, Dat& dat)
	: func(func), dat(dat)
{
}

Dat& TaskGraph::createDat(int n, int m, const std::vector<double> &vals)
{
	std::unique_ptr<Dat> dat(new Dat(n, m, vals));
	dats.push_back(std::move(dat));
	return *dats.back();
}

Task& TaskGraph::createTask(const std::string &func, Dat &dat)
{
	std::unique_ptr<Task> task(new Task(func, dat));
	// TODO: accept dependent tasks as arg, and link
	rootTask = std::move(task);
	return *rootTask;
}

} /* namespace cac */
