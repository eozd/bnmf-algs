cmake_minimum_required(VERSION 3.2.2)
project(bnmf_algs)

set(CMAKE_CXX_STANDARD 14)

include_directories("include")

add_library(bnmf_algs SHARED src/library.cpp include/library.hpp)

add_executable (tests test/test_example.cpp test/include_catch.cpp)
target_link_libraries(tests bnmf_algs)
