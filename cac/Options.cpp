#include "Options.h"

#include <boost/program_options.hpp>
#include <iostream>
#include <cstdlib>

namespace po = boost::program_options;

namespace cac {

class Options::Impl {
public:
	Impl(Options &opts);
	void parse(int argc, char **argv);
public:
	Options &opts;
	po::options_description desc;
	po::variables_map vm;
};

// Defaults
Options::Options()
	: impl(new Options::Impl(*this))
{}

Options::~Options() { }

void Options::parse(int argc, char **argv) {
	impl->parse(argc, argv);
}

void Options::parseOrExit(int argc, char **argv) {
	try {
		impl->parse(argc, argv);
	} catch (std::exception &exc) {
		std::cerr << "ERROR: failed to parse command line arguments: "
			<< exc.what() << std::endl;
		exit(1);
	}
}

Options::Impl::Impl(Options &opts) :
	opts(opts),
	desc("CASPER Meta-program Options")
{
	desc.add_options()
		("help,h", "list supported command line options")
		("profiling-harness", po::bool_switch(&opts.profilingHarness),
		 "generate profiling harness instead of main app")
		("output,o", po::value<std::string>(&opts.llOutputFile)->required(),
		 "name of output file with LLVM IR (.ll)")
		("build-args", po::value<std::string>(&opts.buildArgsFile),
		 "name of output file build arguments for CMake")
		("platform,p", po::value<std::string>(&opts.platformFile)->required(),
		 "name of input file with target platform definition")

		// TODO: These will change eventually: will be generated during the
		// compilation flow, and models are per task, not per app.
		("model", po::value<std::string>(&opts.modelFile)->required(),
		 "name of input file with trained performance prediction model")
		("model-cp", po::value<std::string>(&opts.modelCPFile)->required(),
		 "name of input file with checkpoints for the performance model")
		("candidates", po::value<std::string>(&opts.candidatesFile)->required(),
		 "name of input file with candidates for tunable parameters")
		;
}

void Options::Impl::parse(int argc, char **argv) {
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);
}

} // namespace cac
