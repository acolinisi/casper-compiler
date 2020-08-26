#include "casper.h"

#include <vector>

using namespace cac;

int main(int argc, char **argv) {
	TaskGraph tg("blur"); // must match target name in CMake script
	//Options opts(argc, argv);

	int img_width = 1695, img_height = 1356; // casper.bmp
	//int img_width = 16950, img_height = 13560; // casper-tiled10.bmp
	//int img_width = 33900, img_height = 27120; // casper-tiled20.bmp

	const int BLUR_WIDTH = 16; // also set in Halide generator!

	Dat *img = &tg.createDat(img_height, img_width);

	Task& task_load = tg.createTask(CKernel("bmp_load"), {img});

	Task& task_inv = tg.createTask(CKernel("img_inv"), {img}, {&task_load});

	Dat* img_blurred = &tg.createDat(
			img_height - BLUR_WIDTH - 1,
			img_width - BLUR_WIDTH - 1);

	// TODO: introspect parameters from generator
	Task& task_blur = tg.createTask(HalideKernel("halide_blur",
				{"p1", "p2", "p3", "p4"}),
			{img, img_blurred}, {&task_inv});

	Task& task_save = tg.createTask(CKernel("bmp_save"), {img_blurred});

	return compile(tg);
}
