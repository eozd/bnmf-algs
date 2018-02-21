cmake_minimum_required(VERSION 3.2.2)
project(bnmf_algs)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -g -O0 -fprofile-arcs -ftest-coverage")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgcov")

# Optimized build
# TODO: Use Debug and Release build types instead of commenting
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -O3")

include_directories("include")

find_package(GSL REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${GSL_INCLUDE_DIRS})

add_library(bnmf_algs SHARED src/nmf.cpp include/nmf.hpp include/defs.hpp include/sampling.hpp)
target_link_libraries(bnmf_algs gsl gslcblas)

add_executable (tests test/test_nmf.cpp test/include_catch.cpp)
target_link_libraries(tests bnmf_algs)

#add_executable(dataset_nmf src/main.cpp)
#target_link_libraries(dataset_nmf bnmf_algs)
