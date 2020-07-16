CASPER Compiler
===============

Backend based on MLIR from the LLVM project.

Build
-----

To build the compiler and example apps:

    $ mkdir build && cd build
    $ CC=clang CXX=clang++ cmake -DCMAKE_BUILD_TYPE=Release ..
    $ make
