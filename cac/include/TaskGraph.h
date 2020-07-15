#ifndef CAC_TASK_GRAPH_H
#define CAC_TASK_GRAPH_H

#include <vector>
#include <string>
#include <memory>

namespace cac {

	// Decouple the task graph from compilation implementation:
	// prevent the inclusion of LLVM headers into metaprograms.
	class ValueImpl;
	class ScalarImpl;

	// TODO: move all fields to impl

	class Value {
	public:
		// Want to decouple TaskGraph from implementation of compilation,
		// so can't put compilation functionality into virtual methods.
		enum ValueType {
			Scalar,
			Dat,
		};
	public:
		Value(enum ValueType type, ValueImpl *impl);
	public:
		enum ValueType type;
		ValueImpl *impl;
	};

	class Dat : public Value {
	public:
		Dat(int rows, int cols, const std::vector<double> &vals);
		~Dat();
	public:
		const int rows, cols;
		std::vector<double> vals;
	};

	class Scalar : public Value {
	public:
		// We want to decouple TaskGraph data structure from implementation of
		// compilation, so need to let compiler ask what type objects are (can't
		// put compilation functionality into virtual methods of said objects).
		enum ScalarType {
			Int,
			Float,
		};
	public:
		Scalar(enum ScalarType type, bool initialized, ScalarImpl *impl);
	public:
		enum ScalarType type;
		bool initialized;
		ScalarImpl *impl;
	};

	class IntScalar : public Scalar {
	public:
		IntScalar(uint8_t width);
		IntScalar(uint8_t width, uint64_t v);
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
				std::vector<Value *> args)
			: type(type), func(func), args(args), visited(false) { }
		Task() : args(), visited(false) { }
	public:
		enum TaskType type;
		// TODO: store kernel object instead?
		const std::string func;
		std::vector<Value *> args;
		std::vector<Task *> deps;
		bool visited;
	};

	class HalideTask : public Task {
	public:
		HalideTask(const std::string &func, std::vector<Value *> args)
			: Task(Task::Halide, func, args) {}
	};
	class CTask : public Task {
	public:
		CTask(const std::string &func, std::vector<Value *> args)
			: Task(Task::C, func, args) {}
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
		IntScalar& createIntScalar(uint8_t width);
		IntScalar& createIntScalar(uint8_t width, uint64_t v);

		Task& createTask(HalideKernel kern, std::vector<Value *> args,
				std::vector<Task *> deps = {});
		Task& createTask(CKernel kern, std::vector<Value *> args,
				std::vector<Task *> deps = {});

	protected:
		Task& createTask(std::unique_ptr<Task> task, std::vector<Task *> deps);
		IntScalar& createIntScalar(std::unique_ptr<IntScalar> scalar);

	public:
		std::vector<std::unique_ptr<Value>> values;
		std::vector<std::unique_ptr<Task>> tasks;
	};

} // namespace cac

#endif // CAC_TASK_GRAPH_H
