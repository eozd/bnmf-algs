#!/usr/bin/env bash

# Build the project.
#
# usage: ./build.sh Release
# usage: ./build.sh Debug
#
# If no build type is given, the library is built in Release mode.

build_type=$1

if [[ -z ${build_type} ]]; then
	build_type=Release
fi

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=${build_type} ..
make -j6
