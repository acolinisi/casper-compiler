#ifndef CAC_TASK_GRAPH_H
#define CAC_TASK_GRAPH_H

#include <vector>
#include <string>
#include <memory>

namespace cac {

	class DatImpl;

	class Dat {
	public:
		Dat(int rows, int cols, const std::vector<double> &vals);
		~Dat();
	public:
		const int rows, cols;
		std::vector<double> vals;
		DatImpl *impl;
	};

	class Task {
	public:
		Task(const std::string &func, std::vector<Dat *> dats)
			: func(func), dats(dats), visited(false) { }
		Task() : dats(), visited(false) { }
	public:
		const std::string func;
		std::vector<Dat *> dats;
		std::vector<Task *> deps;
		bool visited;
	};

	class TaskGraph {
	public:
		Dat& createDat(int n, int m, const std::vector<double> &vals);
		Task& createTask(const std::string &func, std::vector<Dat *> dat);
		Task& createTask(const std::string &func, std::vector<Dat *> dat,
				std::vector<Task*> deps);

	public:
		std::vector<std::unique_ptr<Dat>> dats;
		std::vector<std::unique_ptr<Task>> tasks;
	};

} // namespace cac

#endif // CAC_TASK_GRAPH_H
