#include "casper.h"

#include <vector>

using namespace cac;

int main(int argc, char **argv) {
	TaskGraph tg;

	std::vector<double> matVals {
		1.000000e+00, -2.000000e+00, 3.000000e+00,
		4.000000e+00, 5.000000e+00, -6.000000e+00
	};
	Dat& mat = tg.createDat(3, 2, matVals);
	Task& task_dbl = tg.createTask("mat_double", mat);
	Task& task_inv = tg.createTask("mat_invert", mat, {&task_dbl});
	Task& task_abs = tg.createTask("mat_abs", mat, {&task_inv});
	Task& task_dbl2 = tg.createTask("mat_tripple", mat, {&task_dbl});

	Executable exec(tg);
	return exec.emitLLVMIR(); // to stderr
}
