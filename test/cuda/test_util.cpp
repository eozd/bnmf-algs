#include "../catch2.hpp"
#include "cuda/util.hpp"

using namespace bnmf_algs;

TEST_CASE("Test idiv_ceil", "[cuda_util]") {
    size_t a = 5, b = 3;
    REQUIRE(cuda::idiv_ceil(a, b) == 2);

    a = 0, b = 50;
    REQUIRE(cuda::idiv_ceil(a, b) == 0);

    a = 6, b = 3;
    REQUIRE(cuda::idiv_ceil(a, b) == 2);

    a = 6, b = 6;
    REQUIRE(cuda::idiv_ceil(a, b) == 1);

    a = 11, b = 2;
    REQUIRE(cuda::idiv_ceil(a, b) == 6);
}

