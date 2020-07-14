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

	// TODO: Hide this from user
	class Task {
	public:
		// We want to decouple TaskGraph data structure from implementation of
		// compilation, so we have to disambiguate task types explicitly (can't
		// put compilation functionality into virtual methods).
		enum TaskType {
			Halide,
			C
		};

	public:
		Task(enum TaskType type, const std::string &func,
				std::vector<Dat *> dats)
			: type(type), func(func), dats(dats), visited(false) { }
		Task() : dats(), visited(false) { }
	public:
		enum TaskType type;
		// TODO: store kernel object instead?
		const std::string func;
		std::vector<Dat *> dats;
		std::vector<Task *> deps;
		bool visited;
	};

	class HalideTask : public Task {
	public:
		HalideTask(const std::string &func, std::vector<Dat *> dats)
			: Task(Task::Halide, func, dats) {}
	};
	class CTask : public Task {
	public:
		CTask(const std::string &func, std::vector<Dat *> dats)
			: Task(Task::C, func, dats) {}
	};

	class Kernel {
	public:
		Kernel(const std::string &func) : func(func) { }
	public:
		const std::string func;
	};
	class HalideKernel : public Kernel {
	public:
		HalideKernel(const std::string &func) : Kernel(func) { }
	};
	class CKernel : public Kernel {
	public:
		CKernel(const std::string &func) : Kernel(func) { }
	};

	class TaskGraph {
	public:
		Dat& createDat(int n, int m);
		Dat& createDat(int n, int m, const std::vector<double> &vals);

		Task& createTask(HalideKernel kern, std::vector<Dat *> dat,
				std::vector<Task *> deps = {});
		Task& createTask(CKernel kern, std::vector<Dat *> dat,
				std::vector<Task *> deps = {});

	protected:
		Task& createTask(Task *taskp, std::vector<Dat *> dat,
				std::vector<Task *> deps);

	public:
		std::vector<std::unique_ptr<Dat>> dats;
		std::vector<std::unique_ptr<Task>> tasks;
	};

} // namespace cac

#endif // CAC_TASK_GRAPH_H
