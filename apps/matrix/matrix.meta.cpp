#include "casper.h"

#include <vector>

using namespace cac;

int main(int argc, char **argv) {
	TaskGraph tg;

	std::vector<double> matValsA {
		1.000000e+00, -2.000000e+00, 3.000000e+00,
		4.000000e+00, 5.000000e+00, -6.000000e+00
	};
	Dat *matA = &tg.createDat(3, 2, matValsA);

	std::vector<double> matValsB {
		-3.000000e+00, 4.000000e+00, -5.000000e+00,
		6.000000e+00, -7.000000e+00, 8.000000e+00
	};
	Dat* matB = &tg.createDat(3, 2, matValsB);

	//Task& task_dbl = tg.createTask("mat_double", {matA});
	//Task& task_inv = tg.createTask("mat_invert", {matA}, {&task_dbl});
	Task& task_inv = tg.createTask("mat_invert", {matA}, {});
	Task& task_abs = tg.createTask("mat_abs", {matA}, {&task_inv});
	//Task& task_dbl2 = tg.createTask("mat_tripple", {matA}, {&task_dbl});

	Executable exec(tg);
	return exec.emitLLVMIR(); // to stderr
}
