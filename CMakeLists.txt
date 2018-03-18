cmake_minimum_required(VERSION 3.2.2)
project(bnmf_algs)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -pedantic -g -O0 -fprofile-arcs -ftest-coverage")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgcov")

# Optimized build
# TODO: Use Debug and Release build types instead of commenting
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -O3 -fopenmp")

include_directories("include")

find_package(GSL REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
include_directories(SYSTEM ${GSL_INCLUDE_DIRS})

set(LIB_SRC_FILES
        src/nmf/nmf.cpp
        src/allocation_model/sampling.cpp
        src/util/wrappers.cpp
        src/bld/bld_algs.cpp
        src/allocation_model/alloc_model_params.cpp
        src/util/util.cpp
        include/util/generator.hpp)

set(TEST_SRC_FILES
        test/nmf/test_nmf.cpp
        test/include_catch.cpp
        test/util/test_wrappers.cpp
        test/allocation_model/test_sampling.cpp
        test/util/test_util.cpp
        test/bld/test_bld_algs.cpp
        test/allocation_model/test_alloc_model_params.cpp
        test/util/test_generator.cpp)

add_library(bnmf_algs SHARED ${LIB_SRC_FILES})
target_link_libraries(bnmf_algs gsl gslcblas)

add_executable (tests ${TEST_SRC_FILES})
target_link_libraries(tests bnmf_algs)

#add_executable(dataset_nmf src/main_nmf_on_file.cpp)
#target_link_libraries(dataset_nmf bnmf_algs)
