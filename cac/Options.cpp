#include "Options.h"

namespace cac {

// Defaults
Options::Options()
	: kb_filename()
{}

Options::Options(int argc, char **argv) {
		// TODO: use some CLI parser
		kb_filename = argv[1];
		kb_candidates_filename = argv[2];
}

} // namespace cac
