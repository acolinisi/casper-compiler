#include "casper.h"

#include "Executable.h"
#include "KnowledgeBase.h"
#include "Platform.h"
#include "Options.h"

#include "llvm/Support/raw_ostream.h"

namespace {
cac::TuneFunc *tune;

int composeArgsFile(std::ostream &fout,
		cac::TaskGraph &tg, cac::KnowledgeBase &db) {

	// The final application binary is linked using CMake; we (the metaprogram)
	// generate a file with some variable values that are known only by
	// running the metaprogram. CMake loads this file. The variables are:
	//   HALIDE_GENERATORS: names of Halide generators registere in metaprogram
	//   C_KERNEL_SOURCES: list of source files with kernels written in C
	//   NODE_TYPE_IDS: platform hardware description
	fout << "HALIDE_GENERATORS ";
	for (auto &task : tg.tasks) {
		if (task->type == cac::Task::Halide)
			fout << task->func;
	}
	fout << std::endl;

	fout << "NODE_TYPE_IDS ";
	for (auto &nodeType : db.getNodeTypes()) {
		fout << nodeType.id << " ";
	}
	fout << std::endl;
	return 0;
}

} // namespace anon

namespace cac {

// see comments in casper.h
void set_tune_func(TuneFunc &f) {
	tune = &f;
}

int compile(TaskGraph &tg) {

	try {
		std::error_code ec;
		llvm::StringRef outFileName(tg.name + ".ll");
		llvm::raw_fd_ostream fout(outFileName, ec);
		if (ec) {
			std::cerr << "failed to open output file: "
				<< outFileName.str() << ": "
				<< ec.message() << std::endl;
			return 1;
		}

		KnowledgeBase db;
		db.loadPlatform("platform.ini"); // TODO: expose on CLI
		(*tune)(tg, db);

		std::string argsFileName(tg.name + ".args");
		std::ofstream argsFile(argsFileName);
		if (!argsFile.is_open()) {
			std::cerr << "failed to open args output file '"
				<< argsFileName << "': "
				<< strerror(errno) << std::endl;
			return 1;
		}
		int rc = composeArgsFile(argsFile, tg, db);
		if (rc) {
			std::cerr << "failed to compose args file: rc "
			  << rc << std::endl;
			return rc;
		}
		argsFile.close();

		Platform plat{db.getNodeTypes()};
		Executable exec(tg, plat, db);
		return exec.emitLLVMIR(fout);

	} catch (std::exception &exc) {
		std::cerr << "Exception: " << exc.what() << std::endl;
		return 1;
	}
}

} // namespace cac
