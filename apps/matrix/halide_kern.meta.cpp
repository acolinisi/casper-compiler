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
#include <cmath>

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

class HalideBlur : public Halide::Generator<HalideBlur> {
public:
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

        // CPU schedule.
#if 0
        blur_y.split(y, y, yi, 8).parallel(y).vectorize(x, 8);
        blur_x.store_at(blur_y, y).compute_at(blur_y, yi).vectorize(x, 8);
#else
#if 0
        blur_y.split(y, y, yi, 2).parallel(y).vectorize(x, 2);
        blur_x.store_at(blur_y, y).compute_at(blur_y, yi).vectorize(x, 2);
#else
        Var x_i("x_i");
        Var x_i_vi("x_i_vi");
        Var x_i_vo("x_i_vo");
        Var x_o("x_o");
        Var x_vi("x_vi");
        Var x_vo("x_vo");
        Var y_i("y_i");
        Var y_o("y_o");

#if 0
        int power = 10;
        int p1 = rand() % power + 1;
        int p2 = rand() % power + 1;
        int p3 = rand() % p2 + 1; // p2 > p3
        int p4 = rand() % p3 + 1; // p3 > p4
#else
        int p1 = 1;
        int p2 = 1;
        //int p3 = 1 + 1; // p3 > p2
        //int p4 = 1 + 1; // p4 > p3
        int p3 = 1;
        int p4 = 1;
#endif

        int v1 = pow(2,p1);
        int v2 = pow(2,p2);
        int v3 = pow(2,p3);
        int v4 = pow(2,p4);

        {
            Var x = blur_x.args()[0];
            blur_x
                .compute_at(blur_y, x_o)
                .split(x, x_vo, x_vi, v1)
                .vectorize(x_vi);
        }
        {
            Var x = blur_y.args()[0];
            Var y = blur_y.args()[1];
            blur_y
                .compute_root()
                .split(x, x_o, x_i, v2)
                .split(y, y_o, y_i, v3)
                .reorder(x_i, y_i, x_o, y_o)
                .split(x_i, x_i_vo, x_i_vi, v4)
                .vectorize(x_i_vi)
                .parallel(y_o)
                .parallel(x_o);
        }
#endif
#endif

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
