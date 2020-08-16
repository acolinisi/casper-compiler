#include "casper.h"

#include "Executable.h"

namespace cac {

int compile(TaskGraph &tg) {
	Platform plat; // default target is the compilation host
	return compile(tg, plat);
}

int compile(TaskGraph &tg, Platform &plat) {
	Executable exec(tg, plat);
	return exec.emitLLVMIR(); // to stderr
}

} // namespace cac
