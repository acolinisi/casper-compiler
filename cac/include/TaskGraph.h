#ifndef CAC_TASK_GRAPH_H
#define CAC_TASK_GRAPH_H

#include <vector>
#include <string>
#include <memory>

namespace cac {

	class Dat {
	public:
		Dat(int rows, int cols, const std::vector<double> &vals)
		: rows(rows), cols(cols), vals(vals) { }
	public:
		const int rows, cols;
		std::vector<double> vals;
	};

	class Task {
	public:
		Task(const std::string &func, Dat &dat)
			: func(func), dat(&dat), visited(false) { }
		Task() : dat(NULL), visited(false) { }
	public:
		const std::string func;
		Dat *dat; // TODO: multiple
		std::vector<Task *> deps;
		bool visited;
	};

	class TaskGraph {
	public:
		Dat& createDat(int n, int m, const std::vector<double> &vals);
		Task& createTask(const std::string &func, Dat &dat);
		Task& createTask(const std::string &func, Dat &dat,
				std::vector<Task*> deps);

	public:
		std::vector<std::unique_ptr<Dat>> dats;
		std::vector<std::unique_ptr<Task>> tasks;
	};

} // namespace cac

#endif // CAC_TASK_GRAPH_H
