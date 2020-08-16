#include "casper.h"

#include "Executable.h"
#include "KnowledgeBase.h"

namespace cac {

int compile(TaskGraph &tg) {
	Platform plat; // default target is the compilation host
	return compile(tg, plat);
}

int compile(TaskGraph &tg, Platform &plat) {

	// TODO: populate KnowledgeBase through profiling
	// TODO: accept "options" argument to put profiling under a CLI flag
	KnowledgeBase kb;

	Executable exec(tg, plat, kb);
	return exec.emitLLVMIR(); // to stderr
}

} // namespace cac
