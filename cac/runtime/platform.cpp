#include <stdint.h>

extern "C" {

uint32_t get_node_type_id() {
	// TODO read /proc/cpuinfo etc
	// return 0;
	return 1;
}

} // extern C
