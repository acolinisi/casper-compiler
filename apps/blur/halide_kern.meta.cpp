#include "Halide.h"
#include <stdio.h>
#include <cmath>

using namespace Halide;

class HalideBlur : public Halide::Generator<HalideBlur> {
public:
    GeneratorParam<int> tile_x{"tile_x", /* 32 */ 1};  // X tile.
    GeneratorParam<int> tile_y{"tile_y", /* 8 */ 1};   // Y tile.

    Input<Buffer<double>> input{"input", 2};
    Output<Buffer<double>> blur_y{"blur_y", 2};

    static const int BLUR_BOUNDARY = 2; /* depends on pipeline */

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
        int img_width = 1695; // sucks that can't be agnostic, but how else?
        input.dim(0).set_stride(img_width);
        input.dim(1).set_stride(1);
        blur_y.dim(0).set_stride(img_width - BLUR_BOUNDARY);
        blur_y.dim(1).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(HalideBlur, halide_blur)
