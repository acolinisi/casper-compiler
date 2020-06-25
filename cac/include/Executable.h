#ifndef CAC_EXECUTABLE_H
#define CAC_EXECUTABLE_H

#include "TaskGraph.h"

namespace cac {
	class ExecutableImpl;

	class Executable {
	public:
		Executable(TaskGraph &tg);
		~Executable();

		int emitLLVMIR(); // to stderr
		int run();

	private:
		ExecutableImpl *impl;
	};
} // namespace cac

#endif // CAC_EXECUTABLE_H
