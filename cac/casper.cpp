#include "TaskGraph.h"
#include "KnowledgeBase.h"
#include "Platform.h"
#include "Options.h"
#include "InputDesc.h"

#include "tune.h"
#include "halide.h"
#include "mlir.h"
#include "python.h"

#include <Python.h>

#include <fstream>
#include <sstream>
#include <iostream>

namespace {

void composeArgsFile(cac::TaskGraph &tg, cac::KnowledgeBase &db,
		std::vector<std::string>& libs, const std::string &outputFile) {
	std::ofstream fout(outputFile);
	if (!fout.is_open()) {
		std::ostringstream msg;
		msg << "failed to open args output file '"
			<< outputFile << "': " << strerror(errno);
		throw std::runtime_error{msg.str()};
	}

	// The final application binary is linked using CMake; we (the metaprogram)
	// generate a file with some variable values that are known only by
	// running the metaprogram. CMake loads this file. The variables are:
	//   HALIDE_TASK_LIBS: full file names of compiled Halide libraries
	//   C_KERNEL_SOURCES: list of source files with kernels written in C
	//   NODE_TYPE_IDS: platform hardware description
	fout << "HALIDE_TASK_LIBS ";
	for (auto &lib : libs) {
		fout << lib << " ";
	}
	fout << std::endl;

	fout << "NODE_TYPE_IDS ";
	for (auto &nodeType : db.getNodeTypes()) {
		fout << nodeType.id << " ";
	}
	fout << std::endl;
}

void compilePyGeneratedTasks(cac::TaskGraph &tg, const cac::Options &opts) {
	if (tg.pyGenerators.empty()) {
		return;
	}
	cac::py::init(opts.pythonPath);
	for (auto &gen : tg.pyGenerators) {
		PyObject *args[] = {
			PyUnicode_FromString(gen->module.c_str()),
			PyUnicode_FromString(gen->func.c_str()),
		};
		cac::py::launch("pyop2gen", "codegen",
				sizeof(args)/sizeof(args[0]), args, NULL);
		Py_DECREF(args[0]);
		Py_DECREF(args[1]);
	}
	cac::py::finalize();
}

} // namespace anon

namespace cac {

void compile(TaskGraph &tg, const Options &opts) {

	KnowledgeBase db{opts.profilingSamplesFile};
	db.loadPlatform(opts.platformFile);
	cac::Platform plat{db.getNodeTypes()};

	cac::introspectHalideTasks(tg);

	// TODO: list of variant IDs should be per task
	std::vector<unsigned> variantIds;
	std::vector<std::string> halideLibs;
	if (!opts.profilingHarness) {

#if 0
		InputDesc inputDesc(opts.inputDescFile);

		cac::tune(tg, db, inputDesc, opts.modelsDir, opts.candidatesFile);
#else
		db.loadParams(opts.tunedParamsFile);
#endif
		halideLibs = cac::compileHalideTasks(tg, plat, db);

		for (auto &nodeDesc : plat.nodeTypes) {
			variantIds.push_back(nodeDesc.id);
		}
	} else {
		halideLibs = cac::compileHalideTasksToProfile(tg, db);
		for (unsigned vId = 0; vId < db.sampleCount; ++vId) {
			variantIds.push_back(vId);
		}
	}

	compilePyGeneratedTasks(tg, opts);

	if (opts.buildArgsFile.size())
		composeArgsFile(tg, db, halideLibs, opts.buildArgsFile);
	cac::emitLLVMIR(opts.llOutputFile, tg, variantIds,
			opts.profilingHarness, opts.profilingMeasurementsFile);
}

int tryCompile(TaskGraph &tg, const Options &opts) {
	try {
		compile(tg, opts);
		return 0;
	} catch (std::exception &exc) {
		std::cerr << "ERROR: compilation failed with exception: "
			<< exc.what() << std::endl;
		return 1;
	}
}

} // namespace cac
