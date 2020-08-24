#ifndef CAC_EXECUTABLE_H
#define CAC_EXECUTABLE_H

// TODO: private header now: move from cac/include/ to cac/

#include "llvm/Support/raw_ostream.h"

namespace cac {
	class TaskGraph;
	class Platform;
	class KnowledgeBase;

	class ExecutableImpl;

	class Executable {
	public:
		Executable(TaskGraph &tg, Platform &plat, KnowledgeBase &kb);
		~Executable();

		int emitLLVMIR(llvm::raw_ostream &os);
		int run();

	private:
		ExecutableImpl *impl;
	};
} // namespace cac

#endif // CAC_EXECUTABLE_H
