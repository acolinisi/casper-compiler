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
	class PyObjImpl;
	class HalideTaskImpl;
	class PyTaskImpl;

	class Value {
	public:
		Value(ValueImpl *impl);
		virtual ~Value();
		ValueImpl *getImpl();
	protected:
		ValueImpl *impl;
	};

	class Dat : public Value {
	protected:
		Dat(DatImpl *impl);
	public:
		DatImpl *getImpl();
	};
	class DoubleDat : public Dat {
	public:
		DoubleDat(DatImpl *impl) : Dat(impl) {}
	};
	class FloatDat : public Dat {
	public:
		FloatDat(DatImpl *impl) : Dat(impl) {}
	};
	class IntDat : public Dat {
	public:
		IntDat(DatImpl *impl) : Dat(impl) {}
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
	class DoubleScalar : public Scalar {
	public:
		DoubleScalar(double v);
	};

	class PyObj : public Value {
	public:
		PyObj(PyObjImpl *impl);
		PyObjImpl *getImpl();
	};

	// Could hide implementation in impl
	class Task {
	public:
		enum TaskType {
			Halide,
			C,
			Python,
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
		HalideTask(const std::string &func,
			std::vector<Value *> args);
	public:
		std::unique_ptr<HalideTaskImpl> impl;
	};
	class CTask : public Task {
	public:
		CTask(const std::string &func, std::vector<Value *> args)
			: Task(Task::C, func, args) {}
	};
	class PyTask : public Task {
	public:
		enum Type {
			Function,
			Generated
		};

	public:

		PyTask(PyTask::Type type, const std::string &module,
			const std::string &func, std::vector<Value *> args);
		PyTask(const std::string &module, const std::string &func,
			std::vector<Value *> args);
	public:
		enum Type type;
		std::string module;
		std::unique_ptr<PyTaskImpl> impl;
	};
	class PyGenedTask : public PyTask {
	public:
		PyGenedTask(const std::string &module,
				const std::string &kernel,
			std::vector<Value *> args);
	public:
		std::string kernel;
	};

	class Kernel {
	public:
		Kernel(const std::string &func) : func(func) { }
	public:
		const std::string func;
	};
	class HalideKernel : public Kernel {
	public:
		HalideKernel(const std::string &func)
			: Kernel(func) { }
	};
	class CKernel : public Kernel {
	public:
		CKernel(const std::string &func) : Kernel(func) { }
	};
	class PyKernel : public Kernel {
	public:
		PyKernel(const std::string &module, const std::string &func)
			: Kernel(func), module(module) { }
	public:
		std::string module;
	};
	class FEMAKernel : public Kernel {
	public:
		FEMAKernel(const std::string &module,
				const std::string &kernelName)
			: Kernel(kernelName), module(module) { }
	public:
		std::string module;
	};

	class PyGenerator {
		public:
			std::string module;
			std::string func;
	};

	class TaskGraph {
	public:
		TaskGraph(const std::string &name);
		Dat& createDat(int n, int m);
		Dat& createDat(int n, int m, const std::vector<double> &vals);
		Dat& createDoubleDat(int dims, std::vector<int> size);
		Dat& createFloatDat(int dims, std::vector<int> size);
		Dat& createIntDat(int width, int dims, std::vector<int> size);

		PyObj& createPyObj();
		IntScalar& createIntScalar(uint8_t width);
		IntScalar& createIntScalar(uint8_t width, uint64_t v);
		DoubleScalar& createDoubleScalar(double v);

		Task& createTask(HalideKernel kern,
				std::vector<Value *> args = {},
				std::vector<Task *> deps = {});
		Task& createTask(CKernel kern, std::vector<Value *> args = {},
				std::vector<Task *> deps = {});
		Task& createTask(PyKernel kern, std::vector<Value *> args = {},
				std::vector<Task *> deps = {});
		Task& createTask(FEMAKernel kern, std::vector<Value *> args = {},
				std::vector<Task *> deps = {});

		// There's a one-to-many relationship between generators and
		// kernels; for now no association between a generator and the
		// kernels it generates is tracked, but eventually it might
		// make sense to associated generators to tasks.
		void registerPyGenerator(const std::string &mod,
				const std::string &func);

		void setDatPrint(bool enable);

	protected:
		Task& createTask(std::unique_ptr<Task> task, std::vector<Task *> deps);
		IntScalar& createIntScalar(std::unique_ptr<IntScalar> scalar);

	public:
		std::string name;
		std::vector<std::unique_ptr<Value>> values;
		std::vector<std::unique_ptr<Task>> tasks;
		std::vector<std::unique_ptr<PyGenerator>> pyGenerators;
		bool datPrintEnabled;
	};

} // namespace cac

#endif // CAC_TASK_GRAPH_H
