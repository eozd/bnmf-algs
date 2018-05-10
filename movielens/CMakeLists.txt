cmake_minimum_required(VERSION 3.2.2)
project(bnmf_algs_example)

# project and compiler constants
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/../cmake)

# compiler and build flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -O3 -fopenmp")

# add directives
add_definitions(-DNDEBUG)
add_definitions(-DUSE_OPENMP)
add_definitions(-DEIGEN_USE_THREADS)
add_definitions(-DEIGEN_NO_DEBUG)

# include bnmf_algs
include_directories(../src)

# find required libraries
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(GSL REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
include_directories(SYSTEM ${GSL_INCLUDE_DIRS})

# build executable
add_executable(movielens movielens.cpp)
target_link_libraries(movielens gsl gslcblas)
