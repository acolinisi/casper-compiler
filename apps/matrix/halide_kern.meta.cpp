// Halide tutorial lesson 15: Generators part 1

// This lesson demonstrates how to encapsulate Halide pipelines into
// reusable components called generators.

// On linux, you can compile and run it like so:
// g++ lesson_15*.cpp ../tools/GenGen.cpp -g -std=c++11 -fno-rtti -I ../include -L ../bin -lHalide -lpthread -ldl -o lesson_15_generate
// bash lesson_15_generators_usage.sh

// On os x:
// g++ lesson_15*.cpp ../tools/GenGen.cpp -g -std=c++11 -fno-rtti -I ../include -L ../bin -lHalide -o lesson_15_generate
// bash lesson_15_generators_usage.sh

// If you have the entire Halide source tree, you can also build it by
// running:
//    make tutorial_lesson_15_generators
// in a shell with the current directory at the top of the halide
// source tree.

#include "Halide.h"
#include <stdio.h>

using namespace Halide;

// Generators are a more structured way to do ahead-of-time
// compilation of Halide pipelines. Instead of writing an int main()
// with an ad-hoc command-line interface like we did in lesson 10, we
// define a class that inherits from Halide::Generator.
class MyFirstGenerator : public Halide::Generator<MyFirstGenerator> {
public:
    // We declare the Inputs to the Halide pipeline as public
    // member variables. They'll appear in the signature of our generated
    // function in the same order as we declare them.
    Input<uint8_t> offset{"offset"};
    Input<Buffer<double>> input{"input", 2};

    // We also declare the Outputs as public member variables.
    Output<Buffer<double>> brighter{"brighter", 2};

    // Typically you declare your Vars at this scope as well, so that
    // they can be used in any helper methods you add later.
    Var i, j;

    // We then define a method that constructs and return the Halide
    // pipeline:
    void generate() {
        // In lesson 10, here is where we called
        // Func::compile_to_file. In a Generator, we just need to
        // define the Output(s) representing the output of the pipeline.

        // For our 3x2 matrix example, the strides are {2, 1} and the code
        // below will iterate: for(rows){for(columns)}.

        brighter(i, j) = input(i, j) + offset;

        // Schedule it.
        // brighter.vectorize(x, 16).parallel(y);

        // The runtime buffers will be laid out in the same way as the
        // memref objects are (the base type of Casper Dat objects), which
        // defaults to row-major, i.e. matrix[i, j] is element
        // at row i and column j where elements in one row are consecutive in
        // memory. The buffers declared here must match that layout, but
        // Halide's default does not match this memref default, unfortunately
        // -- Halide's default is column-major matrix(x, y) is element in row y
        // column x (which makes sense in context of images and X/Y coordinate
        // system). So, we either need to change layout of the Dat objects
        // declared in the Casper metaprogram, or change the layout of the
        // Halide buffers declared here, we do the latter.

        // TODO: why exactly does Halide compiler need to know the strides at
        // compilation time? At runtime the strides in the passed buffer are
        // checked against the strides specified here at compile time -- but
        // are the strides used at compile time? Having to declare strides
        // here makes the kernel not generic across matrix sizes (at least
        // regarding the last n-1 dimensions).
        input.dim(0).set_stride(4);
        input.dim(1).set_stride(1);
        brighter.dim(0).set_stride(4);
        brighter.dim(1).set_stride(1);
    }
};

enum class BlurGPUSchedule {
    Inline,          // Fully inlining schedule.
    Cache,           // Schedule caching intermedia result of blur_x.
    Slide,           // Schedule enabling sliding window opt within each
                     // work-item or cuda thread.
    SlideVectorize,  // The same as above plus vectorization per work-item.
};

std::map<std::string, BlurGPUSchedule> blurGPUScheduleEnumMap() {
    return {
        {"inline", BlurGPUSchedule::Inline},
        {"cache", BlurGPUSchedule::Cache},
        {"slide", BlurGPUSchedule::Slide},
        {"slide_vector", BlurGPUSchedule::SlideVectorize},
    };
};

class HalideBlur : public Halide::Generator<HalideBlur> {
public:
    GeneratorParam<BlurGPUSchedule> schedule{
        "schedule",
        BlurGPUSchedule::SlideVectorize,
        blurGPUScheduleEnumMap()};
    GeneratorParam<int> tile_x{"tile_x", /* 32 */ 1};  // X tile.
    GeneratorParam<int> tile_y{"tile_y", /* 8 */ 1};   // Y tile.

    Input<Buffer<double>> input{"input", 2};
    Output<Buffer<double>> blur_y{"blur_y", 2};

    void generate() {
        Func blur_x("blur_x");
        Var x("x"), y("y"), xi("xi"), yi("yi");

        // The algorithm
        blur_x(x, y) = (input(x, y) + input(x + 1, y) + input(x + 2, y)) / 3;
        blur_y(x, y) = (blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y + 2)) / 3;

        // How to schedule it
        if (get_target().has_gpu_feature()) {
            // GPU schedule.
            switch (schedule) {
            case BlurGPUSchedule::Inline:
                // - Fully inlining.
                blur_y.gpu_tile(x, y, xi, yi, tile_x, tile_y);
                break;
            case BlurGPUSchedule::Cache:
                // - Cache blur_x calculation.
                blur_y.gpu_tile(x, y, xi, yi, tile_x, tile_y);
                blur_x.compute_at(blur_y, x).gpu_threads(x, y);
                break;
            case BlurGPUSchedule::Slide: {
                // - Instead caching blur_x calculation explicitly, the
                //   alternative is to allow each work-item in OpenCL or thread
                //   in CUDA to calculate more rows of blur_y so that temporary
                //   blur_x calculation is re-used implicitly. This achieves
                //   the similar schedule of sliding window.
                Var y_inner("y_inner");
                blur_y
                    .split(y, y, y_inner, tile_y)
                    .reorder(y_inner, x)
                    .unroll(y_inner)
                    .gpu_tile(x, y, xi, yi, tile_x, 1);
                break;
            }
            case BlurGPUSchedule::SlideVectorize: {
                // Vectorization factor.
                int factor = sizeof(int) / sizeof(short);
                Var y_inner("y_inner");
                blur_y.vectorize(x, factor)
                    .split(y, y, y_inner, tile_y)
                    .reorder(y_inner, x)
                    .unroll(y_inner)
                    .gpu_tile(x, y, xi, yi, tile_x, 1);
                break;
            }
            default:
                break;
            }
        } else if (get_target().features_any_of({Target::HVX_64, Target::HVX_128})) {
            // Hexagon schedule.
            const int vector_size = get_target().has_feature(Target::HVX_128) ? 128 : 64;

            blur_y.compute_root()
                .hexagon()
                .prefetch(input, y, 2)
                .split(y, y, yi, 128)
                .parallel(y)
                .vectorize(x, vector_size * 2);
            blur_x
                .store_at(blur_y, y)
                .compute_at(blur_y, yi)
                .vectorize(x, vector_size);
        } else {
            // CPU schedule.
#if 0
            blur_y.split(y, y, yi, 8).parallel(y).vectorize(x, 8);
            blur_x.store_at(blur_y, y).compute_at(blur_y, yi).vectorize(x, 8);
#endif
        }


        // Match mem layout to Casper Dat buffers (see comments in the other
        // generator)
        input.dim(0).set_stride(4);
        input.dim(1).set_stride(1);
        blur_y.dim(0).set_stride(4 - 2);
        blur_y.dim(1).set_stride(1);
    }
};

// We compile this file along with tools/GenGen.cpp. That file defines
// an "int main(...)" that provides the command-line interface to use
// your generator class. We need to tell that code about our
// generator. We do this like so:
HALIDE_REGISTER_GENERATOR(MyFirstGenerator, halide_bright)
HALIDE_REGISTER_GENERATOR(HalideBlur, halide_blur)
