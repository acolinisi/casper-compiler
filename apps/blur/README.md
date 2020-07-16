Example image-processing application that inverts colors and blurs.
Works on grayscale images in BMP format.

The resulting image will be saved in `apps/blur/casper_blurred.bmp`.

Dimensions are currently hardcoded to correspond to `casper.bmp` test image
(download manually or using `gdown` tool):
  
    $ gdown -O apps/blur/casper.bmp https://drive.google.com/uc?id=1TgfuSwNMFSbzrbFT0-kSkoJlfhz8L088

Build and run (in top-level directory of compiler repo):

    $ cd compiler/
    $ mkdir build && cd build
    $ cmake -DCMAKE_BUILD_TYPE=Release ..
    $ make
    $ cp ../apps/blur/casper.bmp apps/blur/
    $ make blur_run

