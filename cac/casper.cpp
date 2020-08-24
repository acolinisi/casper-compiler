#include "casper.h"

#include "Executable.h"
#include "KnowledgeBase.h"
#include "Platform.h"
#include "Options.h"

namespace cac {

int compile(TaskGraph &tg) {
	Platform plat; // default target is the compilation host
	return compile(tg, plat);
}

int compile(TaskGraph &tg, Platform &plat) {
	Options opts;
	return compile(tg, plat, opts);
}

int compile(TaskGraph &tg, Platform &plat, Options &opts) {
	std::error_code ec;
	llvm::StringRef out_fname(tg.name + ".ll");
	llvm::raw_fd_ostream fout(out_fname, ec);
	if (ec) {
		std::cerr << "failed to open output file: "
			<< out_fname.str() << ": " << ec.message() << std::endl;
		return 1;
	}

	// TODO: populate KnowledgeBase through profiling
	// TODO: accept "options" argument to put profiling under a CLI flag
	KnowledgeBase kb(opts);

	Executable exec(tg, plat, kb);
	return exec.emitLLVMIR(fout);
}

} // namespace cac
