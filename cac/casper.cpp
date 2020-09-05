#include "TaskGraph.h"
#include "KnowledgeBase.h"
#include "Platform.h"
#include "Options.h"

#include "tune.h"
#include "halide.h"
#include "mlir.h"

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

} // namespace anon

namespace cac {

void compile(TaskGraph &tg, const Options &opts) {

	KnowledgeBase db;
	db.loadPlatform(opts.platformFile);
	cac::Platform plat{db.getNodeTypes()};

	cac::introspectHalideTasks(tg);

	std::vector<std::string> halideLibs;
	if (!opts.profilingHarness) {
		cac::tune(tg, db, opts.modelFile, opts.modelCPFile, opts.candidatesFile);
		halideLibs = cac::compileHalideTasks(tg, plat, db);
	} else {
		halideLibs = cac::compileHalideTasksToProfile(tg, db);
	}
	if (opts.buildArgsFile.size())
		composeArgsFile(tg, db, halideLibs, opts.buildArgsFile);
	cac::emitLLVMIR(tg, plat, opts.profilingHarness, opts.llOutputFile);
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
