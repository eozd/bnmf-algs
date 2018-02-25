
#include "catch2.hpp"
#include "decomposition.hpp"
#include "sampling.hpp"
#include "util.hpp"
#include <iostream>

using namespace bnmf_algs;
using namespace bnmf_algs::util;

TEST_CASE("Parameter checks on seq_greedy_bld", "[seq_greedy_bld]") {
    int x = 10, y = 8, z = 3;
    matrix_t X = matrix_t::Random(x, y) + matrix_t::Constant(x, y, 5);
    std::vector<double> alpha(x, 1);
    std::vector<double> beta(z, 1);

    REQUIRE_NOTHROW(seq_greedy_bld(X, z, alpha, beta));

    alpha.resize(x - 1);
    REQUIRE_THROWS(seq_greedy_bld(X, z, alpha, beta));

    alpha.resize(x);
    beta.resize(z - 1);
    REQUIRE_THROWS(seq_greedy_bld(X, z, alpha, beta));

    alpha.resize(x + 1);
    REQUIRE_THROWS(seq_greedy_bld(X, z, alpha, beta));
}

TEST_CASE("Algorithm checks on seq_greedy_bld", "[seq_greedy_bld]") {
    long x = 8, y = 10, z = 3;
    SECTION("Zero matrix") {
        matrix_t X = matrix_t::Zero(x, y);
        std::vector<double> alpha(x, 1.0);
        std::vector<double> beta(z, 1.0);

        tensor3d_t S = seq_greedy_bld(X, z, alpha, beta);

        tensor0d_t min = S.minimum();
        tensor0d_t max = S.maximum();
        REQUIRE(min.coeff() == Approx(0));
        REQUIRE(max.coeff() == Approx(0));
    }

    SECTION("Test against the results of ppmf.py implementation on "
            "Experiments.ipynb") {
        matrix_t X(8, 10);
        X << 0., 1., 0., 0., 0., 3., 2., 0., 2., 2., 10., 10., 15., 21., 15.,
            17., 4., 19., 14., 7., 2., 10., 7., 12., 7., 10., 11., 7., 15., 13.,
            10., 6., 13., 8., 4., 7., 6., 12., 10., 9., 1., 1., 7., 3., 1., 6.,
            0., 5., 2., 8., 2., 3., 4., 1., 7., 13., 4., 1., 2., 4., 5., 3., 5.,
            1., 4., 1., 2., 4., 2., 0., 0., 1., 0., 0., 1., 0., 1., 2., 0., 1.;

        std::vector<double> alpha(x, 1.0);
        std::vector<double> beta(z, 1.0);
        double a = 40, b = 1;

        // todo: how to test if the results are correct?
        tensor3d_t S = seq_greedy_bld(X, z, alpha, beta);
        double log_marginal = log_marginal_S(S, a, b, alpha, beta);

        REQUIRE(log_marginal >= -265);
        REQUIRE(sparseness(S) >= 0.6);
    }
}
