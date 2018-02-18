cmake_minimum_required(VERSION 3.2.2)
project(bnmf_algs)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

include_directories("include")

find_package(GSL REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${GSL_INCLUDE_DIRS})

add_library(bnmf_algs SHARED src/library.cpp include/library.hpp)
add_executable (tests test/test_example.cpp test/include_catch.cpp)

target_link_libraries(bnmf_algs gsl gslcblas)
target_link_libraries(tests bnmf_algs)
