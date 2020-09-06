#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>

#include "halide_benchmark.h"

namespace {

// TODO: is this efficient?
float current_time() {
	auto start_time = Halide::Tools::benchmark_now().time_since_epoch();
	auto now = Halide::Tools::benchmark_now().time_since_epoch() - start_time;
	return std::chrono::duration_cast<std::chrono::microseconds>(now).count()
		/ 1e3;
}

std::ofstream measurements_file;
float start_time;

}

extern "C" {

void _crt_prof_init(const char *measurements_filename) {
	std::cout << "saving profiling measurements to: " << measurements_filename
		<< std::endl;
	measurements_file.open(measurements_filename);
	if (!measurements_file.is_open()) {
		std::cerr << "failed to open measurements output file "
			<< "'" << measurements_filename << "':"
			<< strerror(errno) << std::endl;
		exit(1);
	}
	measurements_file << "task,variant,elapsed_s" << std::endl;
}

void _crt_prof_finalize() {
	measurements_file.close();
}

void _crt_prof_stopwatch_start() {
	start_time = current_time();
	printf("stopwatch start: %f\n", start_time);
}

float _crt_prof_stopwatch_stop() {
	float end_time = current_time();
	printf("stopwatch stop: start %f end %f\n", start_time, end_time);
	return end_time - start_time;
}

void _crt_prof_log_measurement(const char *task, uint32_t variantId,
		float elapsed) {
	measurements_file << task << "," << variantId << "," << elapsed
		<< std::endl;
}

// To do this, would need to generate an adapter for kernel function
// that is a vararg function. So, that we can write the call (to vararg) here,
// otherwise we can't call here because we don't know the function type.
#if 0
typedef void kernel_t(...);
void _crt_prof_invoke_kernel(kernel_t *func, ...) {
}
#endif

} // extern C
