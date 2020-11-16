#include "casper.h"

#include <vector>

using namespace cac;

int main(int argc, char **argv) {
	Options opts; // metaprogram can add custom options
	opts.parseOrExit(argc, argv);

	TaskGraph tg{"ch"};

	tg.registerPyGenerator("kern", "generate");

	PyObj* state = &tg.createPyObj();

	Task& task_init = tg.createTask(PyKernel("kern", "init"), {state});
	Task& task_a_mass = tg.createTask(PyKernel("kern", "assemble_mass"),
			{state}, {&task_init});
	Task& task_a_hats = tg.createTask(PyKernel("kern", "assemble_hats"),
			{state}, {&task_a_mass});
	Task& task_solve = tg.createTask(PyKernel("kern", "solve"), {state},
			{&task_a_hats});
	return tryCompile(tg, opts);
}
