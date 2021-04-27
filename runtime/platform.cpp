#include <cassert>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>

#include <stdint.h>

namespace {
	const std::string RUNTIME_CONFIG_FILE = "crt.ini";
	int variant_id = -1;
}

extern "C" {

void _crt_plat_init() {
	std::ifstream fcfg(RUNTIME_CONFIG_FILE, std::ifstream::in);
	if (!fcfg.good()) {
		std::cerr << "failed to open variant selection file ("
			<< RUNTIME_CONFIG_FILE << "):"
			<< std::strerror(errno) << std::endl;
		exit(1);
	}

	variant_id = -1;
	while (!fcfg.eof()) {
		std::string key, equals;
		fcfg >> key >> equals;
		if (key.compare("variant_id") != 0)
			continue;
		fcfg >> variant_id;
	}
	if (variant_id < 0) {
		std::cerr << "failed to find variant_id option "
			"in variant selection file" << std::endl;
		fcfg.close();
		exit(2);
	}
	fcfg.close();
}

void _crt_plat_finalize() {
}

uint32_t _crt_plat_get_node_type_id() {
	// TODO read /proc/cpuinfo etc
	assert(variant_id >= 0);
	return variant_id;
}

} // extern C
