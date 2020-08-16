#ifndef CAC_EXECUTABLE_H
#define CAC_EXECUTABLE_H

namespace cac {
	class TaskGraph;
	class Platform;
	class KnowledgeBase;

	class ExecutableImpl;

	class Executable {
	public:
		Executable(TaskGraph &tg, Platform &plat, KnowledgeBase &kb);
		~Executable();

		int emitLLVMIR(); // to stderr
		int run();

	private:
		ExecutableImpl *impl;
	};
} // namespace cac

#endif // CAC_EXECUTABLE_H
