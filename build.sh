#!/usr/bin/env bash

# Build the project.
#
# usage: ./build.sh <BUILD_TYPE> <TARGET>
#
# <BUILD_TYPE> must be debug, release, clean.
# <TARGET> can be test, bench.
#
# To pass additional CMake arguments to the build procedure, edit build.config
# file.
#
# example usages:
# 
# 1. Build only the library
#
#        ./build.sh release
#
# 2. Build tests
#
#        ./build.sh release test
#
# 3. Build benchmarks
#
#        ./build.sh release bench
#
# If no build type is given, the library is built in Release mode.

build=$1
shift
target=$1
shift

if [[ -z ${build_type} ]]; then
	build_type=Release
fi

if [[ ${build} == "debug" ]]; then
	build="-DCMAKE_BUILD_TYPE=Debug"
elif [[ ${build} == "release" ]]; then
	build="-DCMAKE_BUILD_TYPE=Release"
elif [[ ${build} == "clean" ]]; then
	rm -rf build/CMakeFiles ||:
	rm -rf build/CMakeCache.txt ||:
	rm -rf build/cmake_install.cmake ||:
	rm -rf build/Makefile ||:

	rm -rf build/dataset_nmf ||:
	rm -rf build/libbnmf_algs.so ||:
	rm -rf build/benchmark ||:
	rm -rf build/tests ||:
	exit
else
	echo "Unknown build type: Use one of debug, release, clean"
	exit
fi

if [[ ${target} == "test" ]]; then
	target="-DBUILD_TESTING=ON"
elif [[ ${target} == "bench" ]]; then
	target="-DBUILD_BENCH=ON"
elif [[ ! -z ${target} ]]; then
	echo "Unknown target: Use one of test, bench"
	exit
fi

additional_flags=`cat build.config | grep -E '-' | tr '\n' ' ' | sed 's/.$//'`
echo 'Following additional flags will be applied:'
echo ${additional_flags} | tr ' ' '\n'
echo

mkdir -p build
cd build
cmake ${build} ${target} ${additional_flags} ..
make -j6
