#!/usr/bin/env bash

mkdir -p build
cd build
cmake .. -DCMAKE_CXX_COMPILER=g++-5 -DCMAKE_C_COMPILER=gcc-5
make -j6

# TODO: CUDA version builds only when make is called twice. What??
make -j6
