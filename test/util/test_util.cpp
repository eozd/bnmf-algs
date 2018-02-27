#include "../catch2.hpp"
#include "util/util.hpp"
#include <iostream>

using namespace bnmf_algs;

int min(int a, int b) { return a < b ? a : b; }

TEST_CASE("Call using std::tuple", "[call]") {
    auto tuple = std::make_tuple(5, 20);
    REQUIRE(util::call(min, tuple) == 5);
}

TEST_CASE("Call using std::array", "[call]") {
    std::array<int, 2> arr{5, 10};
    REQUIRE(util::call(min, arr) == 5);
}

TEST_CASE("Call using std::pair", "[call]") {
    std::pair<int, int> p{5, 10};
    REQUIRE(util::call(min, p) == 5);
}

TEST_CASE("Call using selected elements", "[call]") {
    // Use 0th and 2nd indices
    std::index_sequence<0, 2> indices;
    auto tuple = std::make_tuple(1, "aoesu", 2);
    REQUIRE(util::call(min, tuple, indices) == 1);
}

TEST_CASE("Test sparseness", "[sparseness]") {
    int x = 5, y = 5, z = 5;
    shape<3> tensor_shape{x, y, z};

    SECTION("Zero tensor") {
        tensord<3> S(x, y, z);
        S.setZero();
        REQUIRE(util::sparseness(S) == std::numeric_limits<double>::max());
    }

    SECTION("Tensor with a single nonzero element") {
        tensord<3> S(x, y, z);
        S.setZero();
        S(0, 0, 0) = 240;
        REQUIRE(util::sparseness(S) == Approx(1));
    }

    SECTION("Tensor with all ones") {
        tensord<3> S(x, y, z);
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                for (int k = 0; k < z; ++k) {
                    S(i, j, k) = 1;
                }
            }
        }
        double result = util::sparseness(S);
        REQUIRE(Approx(result).margin(std::numeric_limits<double>::epsilon()) ==
                0);
    }
}
