#include "TaskGraph.h"

#include <vector>

using namespace cac;

int main(int argc, char **argv) {
  cac::TaskGraph tg;

	std::vector<double> matVals {
		1.000000e+00, -2.000000e+00, 3.000000e+00,
		4.000000e+00, 5.000000e+00, -6.000000e+00
	};
	Dat& mat = tg.createDat(3, 2, matVals);
	Task& task = tg.createTask("mat_abs", mat);

	return emitLLVMIR(tg); // to stderr
}
