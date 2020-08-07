#include "casper.h"

#include <vector>

using namespace cac;

int main(int argc, char **argv) {
	TaskGraph tg;

	// TODO: filename + func name?
	Task& task_py = tg.createTask(PyKernel("kern", "solve_cavity"));

	Executable exec(tg);
	return exec.emitLLVMIR(); // to stderr
}
