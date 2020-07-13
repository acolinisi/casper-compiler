#include "casper.h"

#include <vector>

using namespace cac;

int main(int argc, char **argv) {
	TaskGraph tg;

	std::vector<double> matValsA {
		1.000000e+00, -2.000000e+00,
		3.000000e+00, 4.000000e+00,
		5.000000e+00, -6.000000e+00
	};
	Dat *matA = &tg.createDat(3, 2, matValsA);

#if 0
	std::vector<double> matValsB {
		-3.000000e+00, 4.000000e+00,
		-5.000000e+00, 6.000000e+00,
		-7.000000e+00, 8.000000e+00
	};
	Dat* matB = &tg.createDat(3, 2, matValsB);
#endif

#if 0
	Task& task_inv = tg.createTask("mat_invert", {matA}, {});
	Task& task_abs = tg.createTask("mat_abs", {matB}, {});

	Task& task_add = tg.createTask("mat_add", {matA, matB}, {&task_inv, &task_abs});
#else
	// TODO: Halide object
	Dat *matC = &tg.createDat(3, 2);
	Task& task_inv = tg.createTask("halide_bright", {matA, matC}, {});
#endif

	Executable exec(tg);
	return exec.emitLLVMIR(); // to stderr
}
