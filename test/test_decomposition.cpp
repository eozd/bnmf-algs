
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
    AllocModelParams model_params(1, 1, alpha, beta);

    REQUIRE_NOTHROW(seq_greedy_bld(X, z, model_params));

    model_params.alpha.resize(x - 1);
    REQUIRE_THROWS(seq_greedy_bld(X, z, model_params));

    model_params.alpha.resize(x);
    model_params.beta.resize(z - 1);
    REQUIRE_THROWS(seq_greedy_bld(X, z, model_params));

    model_params.alpha.resize(x + 1);
    REQUIRE_THROWS(seq_greedy_bld(X, z, model_params));
}

TEST_CASE("Algorithm checks on seq_greedy_bld", "[seq_greedy_bld]") {
    long x = 8, y = 10, z = 3;
    shape<3> tensor_shape{x, y, z};
    SECTION("Zero matrix") {
        matrix_t X = matrix_t::Zero(x, y);
        AllocModelParams model_params(tensor_shape);

        tensor3d_t S = seq_greedy_bld(X, z, model_params);

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

        AllocModelParams model_params(40, 1, std::vector<double>(x, 1.0),
                                      std::vector<double>(z, 1.0));

        // todo: how to test if the results are correct?
        tensor3d_t S = seq_greedy_bld(X, z, model_params);
        double log_marginal = log_marginal_S(S, model_params);

        REQUIRE(log_marginal >= -265);
        REQUIRE(sparseness(S) >= 0.6);
    }
}

TEST_CASE("Parameter checks on bld_fact", "[bld_fact]") {
    int x = 5, y = 8, z = 2;
    shape<3> tensor_shape{x, y, z};
    tensor3d_t S(x, y, z);
    S.setRandom();
    AllocModelParams model_params(tensor_shape);

    REQUIRE_NOTHROW(bld_fact(S, model_params));

    model_params.alpha.resize(x + 1);
    REQUIRE_THROWS(bld_fact(S, model_params));

    model_params.alpha.resize(x);
    model_params.beta.resize(z + 1);
    REQUIRE_THROWS(bld_fact(S, model_params));

    model_params.alpha.resize(x + 1);
    REQUIRE_THROWS(bld_fact(S, model_params));
}

TEST_CASE("Algorithm checks on bld_fact", "[bld_fact]") {
    int x = 8, y = 10, z = 3;
    shape<3> tensor_shape{x, y, z};
    AllocModelParams model_params(tensor_shape);
    model_params.a = 40;
    model_params.b = 1;
    tensor3d_t S(x, y, z);
    // S(:, :, 0)

    S.setValues({{{3., 0., 0.},
                  {1., 0., 0.},
                  {1., 0., 1.},
                  {1., 0., 3.},
                  {0., 0., 0.},
                  {3., 0., 0.},
                  {0., 0., 2.},
                  {1., 0., 0.},
                  {2., 0., 0.},
                  {0., 0., 2.}},

                 {{2., 0., 0.},
                  {5., 0., 0.},
                  {9., 0., 0.},
                  {5., 0., 1.},
                  {6., 0., 0.},
                  {7., 0., 0.},
                  {9., 2., 0.},
                  {6., 0., 0.},
                  {4., 0., 0.},
                  {2., 0., 0.}},

                 {{0., 0., 3.},
                  {0., 0., 0.},
                  {0., 0., 2.},
                  {0., 0., 4.},
                  {0., 0., 3.},
                  {0., 0., 3.},
                  {0., 1., 3.},
                  {0., 0., 1.},
                  {0., 0., 1.},
                  {0., 0., 2.}},

                 {{0., 2., 0.},
                  {0., 6., 0.},
                  {0., 2., 0.},
                  {0., 4., 0.},
                  {0., 5., 0.},
                  {0., 3., 0.},
                  {0., 7., 0.},
                  {0., 1., 0.},
                  {0., 2., 0.},
                  {0., 3., 0.}},

                 {{27., 0., 0.},
                  {26., 0., 0.},
                  {19., 0., 0.},
                  {3., 0., 0.},
                  {14., 0., 0.},
                  {23., 0., 0.},
                  {13., 0., 0.},
                  {15., 0., 0.},
                  {13., 0., 0.},
                  {4., 0., 0.}},

                 {{0., 0., 1.},
                  {0., 0., 1.},
                  {0., 0., 3.},
                  {0., 0., 7.},
                  {0., 0., 1.},
                  {0., 0., 0.},
                  {0., 3., 5.},
                  {0., 0., 1.},
                  {0., 0., 0.},
                  {0., 0., 6.}},

                 {{0., 0., 3.},
                  {0., 0., 7.},
                  {0., 0., 3.},
                  {0., 0., 14.},
                  {0., 0., 3.},
                  {0., 0., 2.},
                  {0., 0., 14.},
                  {0., 0., 4.},
                  {0., 0., 1.},
                  {0., 0., 12.}},

                 {{2., 0., 0.},
                  {2., 0., 0.},
                  {2., 0., 0.},
                  {2., 0., 0.},
                  {3., 0., 0.},
                  {4., 0., 0.},
                  {0., 1., 0.},
                  {2., 0., 0.},
                  {3., 0., 0.},
                  {0., 0., 0.}}});

    matrix_t W_true(x, z);
    matrix_t H_true(z, y);
    vector_t L_true(y);

    W_true << 0.04918033, 0., 0.06722689, 0.22540984, 0.04761905, 0.00840336,
        0., 0.02380952, 0.18487395, 0., 0.83333333, 0., 0.64344262, 0., 0., 0.,
        0.07142857, 0.21008403, 0., 0., 0.52941176, 0.08196721, 0.02380952, 0.;

    H_true << 0.79069767, 0.70833333, 0.73809524, 0.25, 0.65714286, 0.82222222,
        0.36666667, 0.77419355, 0.84615385, 0.19354839, 0.04651163, 0.125,
        0.04761905, 0.09090909, 0.14285714, 0.06666667, 0.23333333, 0.03225806,
        0.07692308, 0.09677419, 0.1627907, 0.16666667, 0.21428571, 0.65909091,
        0.2, 0.11111111, 0.4, 0.19354839, 0.07692308, 0.70967742;

    L_true << 41., 43.5, 40.5, 41.5, 37., 42., 49.5, 35., 32.5, 35.;

    matrix_t W, H;
    vector_t L;
    std::tie(W, H, L) = bld_fact(S, model_params);

    // Use 8 digit precision since true values are entered with 8 digit
    // precision
    REQUIRE(W.isApprox(W_true, 1e-8));
    REQUIRE(H.isApprox(H_true, 1e-8));
    REQUIRE(L.isApprox(L_true, 1e-8));
}
