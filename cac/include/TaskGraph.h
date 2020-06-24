#ifndef CAC_TASK_GRAPH_H
#define CAC_TASK_GRAPH_H

#include <vector>
#include <string>
#include <memory>

namespace cac {

	class Dat {
	public:
		Dat(int rows, int cols, const std::vector<double> &vals);
	public:
		const int rows, cols;
		std::vector<double> vals;
	};

	class Task {
	public:
		Task(const std::string &func, Dat &dat);
	public:
		const std::string func;
		Dat &dat; // TODO: multiple
	};

	class TaskGraph {
	public:
		Dat& createDat(int n, int m, const std::vector<double> &vals);
		Task& createTask(const std::string &func, Dat &dat);

		Task& root() { return *rootTask; }
	private:
		std::vector<std::unique_ptr<Dat>> dats;
		std::unique_ptr<Task> rootTask;
	};


} /* namespace cac */

#endif /* CAC_TASK_GRAPH_H */
