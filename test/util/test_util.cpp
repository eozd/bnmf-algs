#include "../catch2.hpp"
#include "util/util.hpp"
#include <iostream>

using namespace bnmf_algs::util;

int min(int a, int b) { return a < b ? a : b; }

TEST_CASE("Call using std::tuple", "[util::call]") {
    auto tuple = std::make_tuple(5, 20);
    REQUIRE(call(min, tuple) == 5);
}

TEST_CASE("Call using std::array", "[util::call]") {
    std::array<int, 2> arr{5, 10};
    REQUIRE(call(min, arr) == 5);
}

TEST_CASE("Call using std::pair", "[util::call]") {
    std::pair<int, int> p{5, 10};
    REQUIRE(call(min, p) == 5);
}

TEST_CASE("Call using selected elements", "[util::call]") {
    // Use 0th and 2nd indices
    std::index_sequence<0, 2> indices;
    auto tuple = std::make_tuple(1, "aoesu", 2);
    REQUIRE(call(min, tuple, indices) == 1);
}
