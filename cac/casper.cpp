#include "casper.h"

#include "Executable.h"
#include "KnowledgeBase.h"
#include "Platform.h"
#include "Options.h"

#include "llvm/Support/raw_ostream.h"

namespace {
cac::TuneFunc *tune;
} // namespace anon

namespace cac {

// see comments in casper.h
void set_tune_func(TuneFunc &f) {
	tune = &f;
}

int compile(TaskGraph &tg) {
	Platform plat; // default target is the compilation host
	return compile(tg, plat);
}

int compile(TaskGraph &tg, Platform &plat) {

	try {
		std::error_code ec;
		llvm::StringRef out_fname(tg.name + ".ll");
		llvm::raw_fd_ostream fout(out_fname, ec);
		if (ec) {
			std::cerr << "failed to open output file: "
				<< out_fname.str() << ": " << ec.message() << std::endl;
			return 1;
		}

		KnowledgeBase db;
		(*tune)(tg, db);

		Executable exec(tg, plat, db);
		return exec.emitLLVMIR(fout);

	} catch (std::exception &exc) {
		std::cerr << "Exception: " << exc.what() << std::endl;
		return 1;
	}
}

} // namespace cac
