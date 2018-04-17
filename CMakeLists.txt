cmake_minimum_required(VERSION 3.2.2)
project(bnmf_algs)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -Wall -pedantic -g -O0 -fprofile-arcs -ftest-coverage")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -Wall -pedantic -O3")

if (BUILD_OPENMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    add_definitions(-DUSE_OPENMP)
    add_definitions(-DEIGEN_USE_THREADS)
endif (BUILD_OPENMP)

if (CMAKE_BUILD_TYPE MATCHES Release)
    add_definitions(-DEIGEN_NO_DEBUG)
endif (CMAKE_BUILD_TYPE MATCHES Release)

include_directories("include")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(GSL REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
include_directories(SYSTEM ${GSL_INCLUDE_DIRS})

if (BUILD_CUDA)
    # we are using CUDA 8 ---> the newest possible g++ is g++-5.
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O3")

    add_definitions(-DUSE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})

    add_definitions(-DEIGEN_USE_GPU)
    CUDA_INCLUDE_DIRECTORIES(include ${EIGEN3_INCLUDE_DIR})
endif (BUILD_CUDA)

set(LIB_SRC_FILES
    src/util/wrappers.cpp
    src/bld/bld_algs.cpp
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
    src/cuda/tensor_ops.cu
    src/cuda/tensor_ops_kernels.cu)

set(CUDA_TEST_SRC_FILES
    test/cuda/test_memory.cpp
    test/cuda/test_tensor_ops.cpp)

add_library(bnmf_algs SHARED ${LIB_SRC_FILES})
if (BUILD_CUDA)
    target_link_libraries(bnmf_algs gsl gslcblas ${CUDA_LIBRARIES})
else (BUILD_CUDA)
    target_link_libraries(bnmf_algs gsl gslcblas)
endif (BUILD_CUDA)

# Build cuda code
if (BUILD_CUDA)
    CUDA_COMPILE(CUDA_OBJ_FILES ${CUDA_KERNEL_FILES})
endif (BUILD_CUDA)

# Build tests
if (BUILD_TESTING)
    add_definitions(-DBUILD_TESTING)
    if (BUILD_CUDA)
        add_executable (tests ${TEST_SRC_FILES} ${CUDA_TEST_SRC_FILES} ${CUDA_OBJ_FILES})
        target_link_libraries(tests bnmf_algs ${CUDA_LIBRARIES})
    else (BUILD_CUDA)
        add_executable (tests ${TEST_SRC_FILES})
        target_link_libraries(tests bnmf_algs)
    endif (BUILD_CUDA)
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
