#ifndef CAC_TASK_GRAPH_H
#define CAC_TASK_GRAPH_H

#include <vector>
#include <string>
#include <memory>

namespace cac {

	// Decouple the task graph from compilation implementation:
	// prevent the inclusion of LLVM headers into metaprograms.
	// We could add accessors to the Value types for the user (that would
	// forward to the impl objects), but not needed so far.
	// All Value objects should be constructed only via factory methods in
	// TaskGraph class.
	class ValueImpl;
	class ScalarImpl;
	class DatImpl;

	class Value {
	public:
		Value(ValueImpl *impl);
		virtual ~Value();
		ValueImpl *getImpl();
	protected:
		ValueImpl *impl;
	};

	class Dat : public Value {
	public:
		Dat(DatImpl *impl);
		DatImpl *getImpl();
	};

	class Scalar : public Value {
	public:
		Scalar(ScalarImpl *impl);
		ScalarImpl *getImpl();
	};

	class IntScalar : public Scalar {
	public:
		IntScalar(uint8_t width);
		IntScalar(uint8_t width, uint64_t v);
	};

	// Could hide implementation in impl
	class Task {
	public:
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
		TaskGraph();
		Dat& createDat(int n, int m);
		Dat& createDat(int n, int m, const std::vector<double> &vals);
		IntScalar& createIntScalar(uint8_t width);
		IntScalar& createIntScalar(uint8_t width, uint64_t v);

		Task& createTask(HalideKernel kern, std::vector<Value *> args,
				std::vector<Task *> deps = {});
		Task& createTask(CKernel kern, std::vector<Value *> args,
				std::vector<Task *> deps = {});

		void setDatPrint(bool enable);

	protected:
		Task& createTask(std::unique_ptr<Task> task, std::vector<Task *> deps);
		IntScalar& createIntScalar(std::unique_ptr<IntScalar> scalar);

	public:
		std::vector<std::unique_ptr<Value>> values;
		std::vector<std::unique_ptr<Task>> tasks;
		bool datPrintEnabled;
	};

} // namespace cac

#endif // CAC_TASK_GRAPH_H
