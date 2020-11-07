#include "Options.h"

#include <boost/program_options.hpp>
#include <iostream>
#include <cstdlib>

namespace po = boost::program_options;

namespace {

} // anon namespace

namespace cac {

class Options::Impl {
public:
	Impl(Options &opts);
	void parse(int argc, char **argv);

	bool isStrArgSet(const std::string &arg) {
		return vm.count(arg) && vm[arg].as<std::string>().size() > 0;
	}
	bool isBoolArgSet(const std::string &arg) {
		return vm.count(arg) && vm[arg].as<bool>();
	}

	void requireStrArg(const std::string &arg) {
		if (!isStrArgSet(arg)) {
			std::ostringstream msg;
			msg << "invalid arguments: --" << arg << " is missing";
			throw std::runtime_error(msg.str());
		}
	}
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
		("output,o", po::value<std::string>(&opts.llOutputFile)->required(),
		 "name of output file with LLVM IR (.ll)")
		("build-args", po::value<std::string>(&opts.buildArgsFile),
		 "name of output file build arguments for CMake")
		("platform,p", po::value<std::string>(&opts.platformFile)->required(),
		 "name of input file with target platform definition")
		("input", po::value<std::string>(&opts.inputDescFile),
		 "name of input file with description of input for which to tune")

		("profiling-harness", po::bool_switch(&opts.profilingHarness),
		 "generate profiling harness instead of main app")
		("profiling-measurements",
		 po::value<std::string>(&opts.profilingMeasurementsFile),
		 "name of output file where to save profiling data")
		("profiling-samples",
		 po::value<std::string>(&opts.profilingSamplesFile),
		 "name of output file where to save samples chosen for profiling")

		("models", po::value<std::string>(&opts.modelsDir),
		 "name of input directory with trained performance models")

		("tuned-params", po::value<std::string>(&opts.tunedParamsFile),
		 "name of input file with tuned parameter choices for each variant of each task (INI)")

		// TODO: These will change eventually: will be generated during the
		// compilation flow, and models are per task, not per app.
		("candidates", po::value<std::string>(&opts.candidatesFile),
		 "name of input file with candidates for tunable parameters")
		("python-path", po::value<std::string>(&opts.pythonPath),
		 "Python package search path to append to PYTHONPATH")
		;
}

void Options::Impl::parse(int argc, char **argv) {
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (isBoolArgSet("profiling-harness")) {
		requireStrArg("profiling-measurements");
		requireStrArg("profiling-samples");
	} else { // not profiling-harness
#if 0
		requireStrArg("models");
		requireStrArg("input");
#endif
	}
}

} // namespace cac
