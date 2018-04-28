cmake_minimum_required(VERSION 3.2.2)
project(bnmf_algs_example)

# project and compiler constants
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/../cmake)

# compiler and build flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -O3 -fopenmp")

# definitions used by bnmf_algs library
add_definitions(-DUSE_OPENMP)
add_definitions(-DUSE_CUDA)
add_definitions(-DEIGEN_USE_THREADS)
add_definitions(-DEIGEN_NO_DEBUG)
add_definitions(-DEIGEN_USE_GPU)

# include bnmf_algs
include_directories(../src)

# find required libraries
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(GSL REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
include_directories(SYSTEM ${GSL_INCLUDE_DIRS})

# CUDA kernel files directory
set(CUDA_SRC_DIR "../cuda_src")

# CUDA source files
set(CUDA_KERNEL_FILES
	${CUDA_SRC_DIR}/tensor_ops.cu
	${CUDA_SRC_DIR}/tensor_ops_kernels.cu
	${CUDA_SRC_DIR}/bld_mult/bld_mult_cuda_funcs.cu
	${CUDA_SRC_DIR}/bld_mult/bld_mult_cuda_kernels.cu)

# CUDA options and directories
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_AVOID_STL_ARRAY")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11;-O3;--maxrregcount=32;-DEIGEN_AVOID_STL_ARRAY")
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
CUDA_INCLUDE_DIRECTORIES(src ${EIGEN3_INCLUDE_DIR})

# build executable
CUDA_ADD_EXECUTABLE(main_file main_file_factorization.cpp ${CUDA_KERNEL_FILES})
target_link_libraries(main_file gsl gslcblas ${CUDA_LIBRARIES})
