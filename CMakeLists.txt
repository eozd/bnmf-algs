cmake_minimum_required(VERSION 3.2.2)
project(bnmf_algs)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -Wall -pedantic -g -fprofile-arcs -ftest-coverage")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -Wall -pedantic -O3")

if (BUILD_OPENMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    add_definitions(-DUSE_OPENMP)
    add_definitions(-DEIGEN_USE_THREADS)
endif (BUILD_OPENMP)

if (CMAKE_BUILD_TYPE MATCHES Release)
    add_definitions(-DEIGEN_NO_DEBUG)
endif (CMAKE_BUILD_TYPE MATCHES Release)

include_directories(src)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(GSL REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
include_directories(SYSTEM ${GSL_INCLUDE_DIRS})

set(TEST_SRC_FILES
    test/nmf/test_nmf.cpp
    test/include_catch.cpp
    test/allocation_model/test_alloc_model_funcs.cpp
    test/util/test_util.cpp
    test/bld/test_bld_algs.cpp
    test/bld/bld_mult/test_bld_mult.cpp
    test/allocation_model/test_alloc_model_params.cpp
    test/util/test_generator.cpp
    test/util/test_sampling.cpp)

set(BENCH_SRC_FILES
    benchmark/benchmark.cpp)

set(CUDA_KERNEL_FILES
    cuda_src/tensor_ops.cu
    cuda_src/tensor_ops_kernels.cu
    cuda_src/bld_mult/bld_mult_cuda_funcs.cu
    cuda_src/bld_mult/bld_mult_cuda_kernels.cu)

set(CUDA_TEST_SRC_FILES
    test/cuda/test_memory.cpp
    test/cuda/test_tensor_ops.cpp
    test/cuda/test_util.cpp
    test/bld/bld_mult/test_bld_mult_cuda.cpp
    test/bld/bld_mult/test_bld_mult_cuda_funcs.cpp)

if (BUILD_CUDA)
    # we are using CUDA 8 ---> the newest possible g++ is g++-5.
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    set(CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_AVOID_STL_ARRAY")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11;-O3;--maxrregcount=32;-DEIGEN_AVOID_STL_ARRAY")

    add_definitions(-DUSE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})

    add_definitions(-DEIGEN_USE_GPU)
    CUDA_INCLUDE_DIRECTORIES(src ${EIGEN3_INCLUDE_DIR})
endif (BUILD_CUDA)

# Build tests
if (BUILD_TESTING)
    add_definitions(-DBUILD_TESTING)
    if (BUILD_CUDA)
        CUDA_ADD_EXECUTABLE(tests ${TEST_SRC_FILES} ${CUDA_TEST_SRC_FILES} ${CUDA_KERNEL_FILES})
        target_link_libraries(tests ${CUDA_LIBRARIES})
    else (BUILD_CUDA)
        add_executable (tests ${TEST_SRC_FILES})
        target_link_libraries(tests)
    endif (BUILD_CUDA)
    target_link_libraries(tests gsl gslcblas)
endif (BUILD_TESTING)

# Build benchmarks
if (BUILD_BENCH)
    # todo: move this elsewhere?
    if (BUILD_CUDA)
        CUDA_ADD_EXECUTABLE(main_file main_file_factorization.cpp ${CUDA_KERNEL_FILES})
        CUDA_ADD_EXECUTABLE(benchmark ${BENCH_SRC_FILES} ${CUDA_KERNEL_FILES})

        target_link_libraries(main_file gsl gslcblas ${CUDA_LIBRARIES})
        target_link_libraries(benchmark ${CUDA_LIBRARIES})
    else (BUILD_CUDA)
        add_executable(main_file main_file_factorization.cpp)
        add_executable(benchmark ${BENCH_SRC_FILES})

        target_link_libraries(main_file gsl gslcblas)
        target_include_directories(benchmark PUBLIC /usr/local/include/celero)
    endif (BUILD_CUDA)
    target_link_libraries(benchmark /usr/local/lib/libcelero.so Threads::Threads gsl gslcblas)
endif (BUILD_BENCH)
