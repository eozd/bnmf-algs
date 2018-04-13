cmake_minimum_required(VERSION 3.2.2)
project(bnmf_algs)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -Wall -pedantic -g -O0 -fprofile-arcs -ftest-coverage")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -Wall -pedantic -O3")

if (OPENMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    add_definitions(-DUSE_OPENMP)
endif (OPENMP)

if (CMAKE_BUILD_TYPE MATCHES RELEASE)
    add_definitions(-DNDEBUG)
endif (CMAKE_BUILD_TYPE MATCHES RELEASE)

include_directories("include")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(GSL REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
include_directories(SYSTEM ${GSL_INCLUDE_DIRS})

if (BUILD_CUDA)
    # we are using CUDA 8 ---> the newest possible g++ is g++-5.
    set(CMAKE_CXX_COMPILER g++-5)
    add_definitions(-DUSE_CUDA)
    find_package(CUDA REQUIRED)
    set(CUDA_HOST_COMPILER g++-5)
    set(CUDA_NVCC_FLAGS "-std=c++11")
    CUDA_INCLUDE_DIRECTORIES(include ${EIGEN3_INCLUDE_DIR})
endif (BUILD_CUDA)

set(LIB_SRC_FILES
    src/nmf/nmf.cpp
    src/allocation_model/alloc_model_funcs.cpp
    src/util/wrappers.cpp
    src/bld/bld_algs.cpp
    src/allocation_model/alloc_model_params.cpp
    src/util/util.cpp
    src/util/sampling.cpp)

set(TEST_SRC_FILES
    test/nmf/test_nmf.cpp
    test/include_catch.cpp
    test/util/test_wrappers.cpp
    test/allocation_model/test_alloc_model_funcs.cpp
    test/util/test_util.cpp
    test/bld/test_bld_algs.cpp
    test/allocation_model/test_alloc_model_params.cpp
    test/util/test_generator.cpp
    test/util/test_sampling.cpp)

set(BENCH_SRC_FILES
    benchmark/benchmark.cpp
    benchmark/bench_utils.cpp)

set(CUDA_KERNEL_FILES
    src/cuda/cuda_kernel.cu)

set(CUDA_SRC_FILES
    src/cuda/main_cuda.cpp)

add_library(bnmf_algs SHARED ${LIB_SRC_FILES})
target_link_libraries(bnmf_algs gsl gslcblas)

# Build tests
if (BUILD_TESTING)
    add_executable (tests ${TEST_SRC_FILES})
    target_link_libraries(tests bnmf_algs)
endif (BUILD_TESTING)

# Build benchmarks
if (BUILD_BENCH)
    # todo: move this elsewhere?
    add_executable(dataset_nmf src/main_nmf_on_file.cpp)
    target_link_libraries(dataset_nmf bnmf_algs)

    add_executable(benchmark ${BENCH_SRC_FILES})
    target_include_directories(benchmark PUBLIC /usr/local/include/celero)
    target_link_libraries(benchmark /usr/local/lib/libcelero.so Threads::Threads)
    target_link_libraries(benchmark bnmf_algs)
endif (BUILD_BENCH)

# Build cuda code
if (BUILD_CUDA)
    CUDA_COMPILE(CUDA_OBJ_FILES ${CUDA_KERNEL_FILES})
    add_executable(cuda_exec ${CUDA_SRC_FILES} ${CUDA_OBJ_FILES})
    target_link_libraries(cuda_exec bnmf_algs ${CUDA_LIBRARIES})
endif (BUILD_CUDA)
