#include "casper.h"

#include <vector>

using namespace cac;

int main(int argc, char **argv) {
	Options opts; // metaprogram can add custom options
	opts.parseOrExit(argc, argv);

	TaskGraph tg{"ch"};

	tg.registerPyGenerator("kern", "generate");

	PyObj* state = &tg.createPyObj();

	Task& task_init = tg.createTask(PyKernel("kern", "init_ch"), {state});
	Task& task_fem = tg.createTask(PyKernel("kern", "solve_ch"), {state},
			{&task_init});
	return tryCompile(tg, opts);
}
