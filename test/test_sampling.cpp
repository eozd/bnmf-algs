#include "sampling.hpp"
#include "catch2.hpp"


using namespace bnmf_algs;

TEST_CASE("sample_S parameter checks", "sample_S") {
    SECTION("Sampling using distribution parameters", "sample_S_parameters") {
        int x = 5, y = 3, z = 2;
        int a = 3, b = 2;
        std::tuple<int, int, int> shape{x, y, z};
        std::vector<double> alpha(x);
        std::vector<double> beta(z);

        REQUIRE_NOTHROW(sample_S(shape, a, b, alpha, beta));

        std::get<0>(shape)++;
        REQUIRE_THROWS(sample_S(shape, a, b, alpha, beta));

        alpha.resize(std::get<0>(shape));
        REQUIRE_NOTHROW(sample_S(shape, a, b, alpha, beta));

        beta.resize(z + 2);
        REQUIRE_THROWS(sample_S(shape, a, b, alpha, beta));

        std::get<2>(shape) += 2;
        std::get<1>(shape) = 0;
        REQUIRE_THROWS(sample_S(shape, a, b, alpha, beta));

        std::get<0>(shape) = -1;
        std::get<1>(shape) = 1;
        REQUIRE_THROWS(sample_S(shape, a, b, alpha, beta));

        std::get<0>(shape) = 2;
        std::get<1>(shape) = 1;
        std::get<2>(shape) = 0;
        REQUIRE_THROWS(sample_S(shape, a, b, alpha, beta));
    }

    SECTION("Sampling using prior matrices", "sample_S_matrices") {
        int x = 10, y = 8, z = 5;
        matrix_t W(x, z);
        matrix_t H(z, y);
        vector_t L(y);

        REQUIRE_NOTHROW(sample_S(W, H, L));

        L = vector_t::Ones(y + 1);
        REQUIRE_THROWS(sample_S(W, H, L));

        H = matrix_t::Ones(z, y + 1);
        W = matrix_t::Ones(x, z + 2);
        REQUIRE_THROWS(sample_S(W, H, L));

        H = matrix_t::Ones(z + 2, y + 1);
        REQUIRE_NOTHROW(sample_S(W, H, L));
    }
}
