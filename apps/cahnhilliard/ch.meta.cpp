#include "casper.h"

#include <vector>

using namespace cac;

int main(int argc, char **argv) {
	Options opts; // metaprogram can add custom options
	opts.parseOrExit(argc, argv);

	TaskGraph tg{"ch"};

	tg.registerPyGenerator("kern", "generate");

	PyObj* sol = &tg.createPyObj();

	Task& task_fem = tg.createTask(PyKernel("kern", "solve_ch"), {sol});
	return tryCompile(tg, opts);
}
