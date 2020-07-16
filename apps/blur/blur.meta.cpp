#include "casper.h"

#include <vector>

using namespace cac;

int main(int argc, char **argv) {
	TaskGraph tg;

	std::vector<double> matValsA {
		1.000000e+00, -2.000000e+00, 3.000000e+00, 4.000000e+00,
		3.000000e+00, 4.000000e+00, 5.000000e+00, -6.000000e+00,
		5.000000e+00, -6.000000e+00, 4.000000e+00, 5.000000e+00,
		-2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00,
		3.000000e+00, 4.000000e+00, 5.000000e+00, -6.000000e+00,
		5.000000e+00, -6.000000e+00, 4.000000e+00, 5.000000e+00,
	};
	Dat *matA = &tg.createDat(6, 4, matValsA);

	Task& task_norm = tg.createTask(CKernel("mat_norm"), {matA});

	Dat* matB = &tg.createDat(6 - 2, 4 - 2);
	Task& task_blur = tg.createTask(HalideKernel("halide_blur"),
			{matA, matB}, {&task_norm});

	Executable exec(tg);
	return exec.emitLLVMIR(); // to stderr
}
