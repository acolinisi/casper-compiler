#include "casper.h"

#include <vector>

using namespace cac;

int main(int argc, char **argv) {
	TaskGraph tg;

	int img_width = 1695, img_height = 1356; // casper.bmp
	Dat *img = &tg.createDat(img_height, img_width);

	Task& task_load = tg.createTask(CKernel("bmp_load"), {img});

	Task& task_inv = tg.createTask(CKernel("img_inv"), {img}, {&task_load});

	Dat* img_blurred = &tg.createDat(img_height - 2, img_width - 2);
	Task& task_blur = tg.createTask(HalideKernel("halide_blur"),
			{img, img_blurred}, {&task_inv});

	Task& task_save = tg.createTask(CKernel("bmp_save"), {img_blurred});

	Executable exec(tg);
	return exec.emitLLVMIR(); // to stderr
}
