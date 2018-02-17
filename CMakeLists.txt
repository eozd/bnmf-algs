cmake_minimum_required(VERSION 3.9)
project(bnmf_algs)

set(CMAKE_CXX_STANDARD 17)

include_directories("include")

add_library(bnmf_algs SHARED src/library.cpp include/library.hpp)