#include "HalideBuffer.h"
#include "halide_bright.h"

#include <stdio.h>

int main(int argc, char **argv) {

	Halide::Runtime::Buffer<double> input(2, 3), output(2, 3);
	int offset = 0;
	struct halide_buffer_t *input_buf =
		static_cast<struct halide_buffer_t *>(input);
	int error = halide_bright(offset, input, output);
	if (error) {
		printf("Halide kernel error: %d\n", error);
		return 1;
	}
	return 0;
}
