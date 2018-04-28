#!/usr/bin/env bash

mkdir -p build
cd build
cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc
make -j6
