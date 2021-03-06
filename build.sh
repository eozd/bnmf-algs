#!/usr/bin/env bash

# Build the project.
#
# usage: ./build.sh <BUILD_TYPE> <TARGET>
#
# <BUILD_TYPE> must be debug, release, clean, doc.
# <TARGET> can be test, bench.
#
# To pass additional CMake arguments to the build procedure, edit build.config
# file.
#
# example usages:
# 
# 1. Build tests
#
#        ./build.sh release test
#
# 2. Build benchmarks
#
#        ./build.sh release bench
#
# 3. Build documentation
#
#        ./build.sh doc

build=$1
shift
target=$1
shift

if [[ ${build} == "debug" ]]; then
	build="-DCMAKE_BUILD_TYPE=Debug"
elif [[ ${build} == "release" ]]; then
	build="-DCMAKE_BUILD_TYPE=Release"
elif [[ ${build} == "clean" ]]; then
	rm -rf build/CMakeFiles ||:
	rm -rf build/CMakeCache.txt ||:
	rm -rf build/cmake_install.cmake ||:
	rm -rf build/Makefile ||:
	rm -rf build/benchmark ||:
	rm -rf build/tests ||:

	rm -rf doc/build ||:
	exit
elif [[ ${build} == "doc" ]]; then
	doxygen doc/Doxyfile
	exit
else
	echo "Unknown build type: Use one of debug, release, clean, doc"
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
