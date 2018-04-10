#include "../catch2.hpp"
#include "allocation_model/alloc_model_funcs.hpp"
#include "bld/bld_algs.hpp"
#include "util/util.hpp"
#include <iostream>

using namespace bnmf_algs;
using namespace bnmf_algs::util;
using namespace bnmf_algs::allocation_model;

TEST_CASE("Parameter checks on seq_greedy_bld", "[seq_greedy_bld]") {
    size_t x = 10, y = 8, z = 3;
    matrix_t X = matrix_t::Random(x, y) + matrix_t::Constant(x, y, 5);
    std::vector<double> alpha(x, 1);
    std::vector<double> beta(z, 1);
    AllocModelParams model_params(1, 1, alpha, beta);

    REQUIRE_NOTHROW(bld::seq_greedy_bld(X, z, model_params));

    model_params.alpha.resize(x - 1);
    REQUIRE_THROWS(bld::seq_greedy_bld(X, z, model_params));

    model_params.alpha.resize(x);
    model_params.beta.resize(z - 1);
    REQUIRE_THROWS(bld::seq_greedy_bld(X, z, model_params));

    model_params.alpha.resize(x + 1);
    REQUIRE_THROWS(bld::seq_greedy_bld(X, z, model_params));

    model_params.alpha.resize(x);
    model_params.beta.resize(z);
    X(0, 0) = -5;
    REQUIRE_THROWS(bld::seq_greedy_bld(X, z, model_params));
}

TEST_CASE("Algorithm checks on seq_greedy_bld", "[seq_greedy_bld]") {
    size_t x = 8, y = 10, z = 3;
    shape<3> tensor_shape{x, y, z};
    SECTION("Zero matrix result is Zero") {
        matrix_t X = matrix_t::Zero(x, y);
        AllocModelParams model_params(tensor_shape);

        tensord<3> S = bld::seq_greedy_bld(X, z, model_params);

        tensord<0> min = S.minimum();
        tensord<0> max = S.maximum();
        REQUIRE(min.coeff() == Approx(0));
        REQUIRE(max.coeff() == Approx(0));
    }

    SECTION("Test against the results of ppmf.py implementation on "
            "Experiments.ipynb") {
        matrix_t X(x, y);
        X << 0., 1., 0., 0., 0., 3., 2., 0., 2., 2., 10., 10., 15., 21., 15.,
            17., 4., 19., 14., 7., 2., 10., 7., 12., 7., 10., 11., 7., 15., 13.,
            10., 6., 13., 8., 4., 7., 6., 12., 10., 9., 1., 1., 7., 3., 1., 6.,
            0., 5., 2., 8., 2., 3., 4., 1., 7., 13., 4., 1., 2., 4., 5., 3., 5.,
            1., 4., 1., 2., 4., 2., 0., 0., 1., 0., 0., 1., 0., 1., 2., 0., 1.;

        AllocModelParams model_params(40, 1, std::vector<double>(x, 1.0),
                                      std::vector<double>(z, 1.0));

        // todo: how to test if the results are correct?
        tensord<3> S = bld::seq_greedy_bld(X, z, model_params);
        double log_marginal = log_marginal_S(S, model_params);

        // Test marginal and sparseness
        REQUIRE(log_marginal >= -270);
        REQUIRE(sparseness(S) >= 0.6);

        // Test if S_{::+} = X
        tensord<2> sum_S = S.sum(shape<1>({2}));
        matrix_t sum_S_mat = Eigen::Map<matrix_t>(
            sum_S.data(), sum_S.dimension(0), sum_S.dimension(1));
        REQUIRE(X.isApprox(sum_S_mat));
        // todo: check if S is an integer tensor
    }
}

TEST_CASE("Parameter checks on bld_fact", "[bld_fact]") {
    size_t x = 5, y = 8, z = 2;
    shape<3> tensor_shape{x, y, z};
    tensord<3> S(x, y, z);
    S.setRandom();
    AllocModelParams model_params(tensor_shape);

    REQUIRE_NOTHROW(bld::bld_fact(S, model_params));

    model_params.alpha.resize(x + 1);
    REQUIRE_THROWS(bld::bld_fact(S, model_params));

    model_params.alpha.resize(x);
    model_params.beta.resize(z + 1);
    REQUIRE_THROWS(bld::bld_fact(S, model_params));

    model_params.alpha.resize(x + 1);
    REQUIRE_THROWS(bld::bld_fact(S, model_params));
}

TEST_CASE("Algorithm checks on bld_fact", "[bld_fact]") {
    size_t x = 8, y = 10, z = 3;
    shape<3> tensor_shape{x, y, z};
    AllocModelParams model_params(tensor_shape);
    model_params.a = 40;
    model_params.b = 1;
    tensord<3> S(x, y, z);
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
    std::tie(W, H, L) = bld::bld_fact(S, model_params);

    // Use 8 digit precision since true values are entered with 8 digit
    // precision
    REQUIRE(W.isApprox(W_true, 1e-8));
    REQUIRE(H.isApprox(H_true, 1e-8));
    REQUIRE(L.isApprox(L_true, 1e-8));
}

TEST_CASE("Parameter checks on bld_mult", "[bld_mult]") {
    size_t x = 10, y = 5, z = 2;
    shape<3> tensor_shape{x, y, z};
    matrix_t X = matrix_t::Constant(x, y, 5);
    allocation_model::AllocModelParams model_params(tensor_shape);

    // max_iter = 5 since we don't want to wait
    REQUIRE_NOTHROW(bld::bld_mult(X, z, model_params, 5));

    X(0, 0) = -5;
    REQUIRE_THROWS(bld::bld_mult(X, z, model_params));

    X(0, 0) = 0;
    REQUIRE_NOTHROW(bld::bld_mult(X, z, model_params, 5));

    z++;
    REQUIRE_THROWS(bld::bld_mult(X, z, model_params));

    z--;
    X = matrix_t::Constant(x + 1, y, 5);
    REQUIRE_THROWS(bld::bld_mult(X, z, model_params));
}

TEST_CASE("Algorithm checks on bld_mult", "[bld_mult]") {
    size_t x = 8, y = 10, z = 3;
    shape<3> tensor_shape{x, y, z};
    SECTION("Zero matrix result is Zero") {
        matrix_t X = matrix_t::Zero(x, y);
        AllocModelParams model_params(tensor_shape);

        tensord<3> S = bld::bld_mult(X, z, model_params);

        tensord<0> min = S.minimum();
        tensord<0> max = S.maximum();
        REQUIRE(min.coeff() == Approx(0));
        REQUIRE(max.coeff() == Approx(0));
    }

    SECTION("Test against the results of ppmf.py implementation on "
            "Experiments.ipynb") {
        matrix_t X(x, y);
        X << 7., 11., 5., 8., 8., 8., 7., 6., 3., 9., 3., 7., 3., 1., 0., 6.,
            1., 0., 1., 4., 1., 11., 4., 5., 3., 8., 5., 3., 3., 3., 0., 20.,
            11., 16., 14., 26., 16., 23., 16., 15., 0., 1., 0., 2., 2., 2., 0.,
            1., 0., 1., 3., 4., 1., 0., 4., 0., 2., 0., 1., 5., 4., 4., 2., 2.,
            3., 4., 0., 2., 0., 5., 4., 3., 3., 1., 2., 6., 5., 7., 4., 2.;

        AllocModelParams model_params(40, 1, std::vector<double>(x, 1.0),
                                      std::vector<double>(z, 1.0));

        // Test with exact digamma
        // =======================
        tensord<3> S = bld::bld_mult(X, z, model_params, 2000);
        double log_marginal = log_marginal_S(S, model_params);
        REQUIRE(log_marginal >= -262);
        REQUIRE(sparseness(S) >= 0.64);

        // Test if S_{::+} = X
        tensord<2> sum_S = S.sum(shape<1>({2}));
        matrix_t sum_S_mat = Eigen::Map<matrix_t>(
            sum_S.data(), sum_S.dimension(0), sum_S.dimension(1));
        REQUIRE(X.isApprox(sum_S_mat, 1e-10));

        // Test with approximate digamma
        // =============================
        S = bld::bld_mult(X, z, model_params, 2000, true);
        log_marginal = log_marginal_S(S, model_params);
        REQUIRE(log_marginal >= -262);
        REQUIRE(sparseness(S) >= 0.64);

        // Test if S_{::+} = X
        sum_S = S.sum(shape<1>({2}));
        sum_S_mat = Eigen::Map<matrix_t>(
                sum_S.data(), sum_S.dimension(0), sum_S.dimension(1));
        REQUIRE(X.isApprox(sum_S_mat, 1e-4));
        // todo: check if S is an integer tensor
    }
}

TEST_CASE("Parameter checks on bld_add", "[bld_add]") {
    size_t x = 10, y = 5, z = 2;
    shape<3> tensor_shape{x, y, z};
    matrix_t X = matrix_t::Constant(x, y, 5);
    allocation_model::AllocModelParams model_params(tensor_shape);

    // max_iter = 5 since we don't want to wait
    REQUIRE_NOTHROW(bld::bld_add(X, z, model_params, 5));

    X(0, 0) = -5;
    REQUIRE_THROWS(bld::bld_add(X, z, model_params));

    X(0, 0) = 0;
    REQUIRE_NOTHROW(bld::bld_add(X, z, model_params, 5));

    z++;
    REQUIRE_THROWS(bld::bld_add(X, z, model_params));

    z--;
    X = matrix_t::Constant(x + 1, y, 5);
    REQUIRE_THROWS(bld::bld_add(X, z, model_params));
}

TEST_CASE("Algorithm checks on bld_add", "[bld_add]") {
    size_t x = 8, y = 10, z = 3;
    shape<3> tensor_shape{x, y, z};
    SECTION("Zero matrix result is Zero") {
        matrix_t X = matrix_t::Zero(x, y);
        AllocModelParams model_params(tensor_shape);

        tensord<3> S = bld::bld_add(X, z, model_params);

        tensord<0> min = S.minimum();
        tensord<0> max = S.maximum();
        REQUIRE(min.coeff() == Approx(0));
        REQUIRE(max.coeff() == Approx(0));
    }

    SECTION("Test against the results of ppmf.py implementation on "
            "Experiments.ipynb") {
        matrix_t X(x, y);
        X << 6., 3., 7., 15., 8., 6., 1., 7., 6., 7., 3., 3., 11., 1., 3., 3.,
            0., 1., 3., 1., 1., 2., 2., 8., 3., 2., 0., 2., 6., 1., 8., 9., 18.,
            5., 8., 6., 19., 6., 11., 8., 6., 3., 5., 1., 2., 0., 2., 4., 0.,
            2., 2., 1., 2., 2., 1., 0., 1., 2., 7., 1., 6., 10., 15., 6., 3.,
            2., 9., 4., 12., 4., 3., 2., 0., 4., 2., 2., 1., 3., 2., 2.;

        AllocModelParams model_params(40, 1, std::vector<double>(x, 1.0),
                                      std::vector<double>(z, 1.0));

        // todo: how to test if the results are correct?
        tensord<3> S = bld::bld_add(X, z, model_params);
        double log_marginal = log_marginal_S(S, model_params);

        REQUIRE(log_marginal >= -273);
        REQUIRE(sparseness(S) >= 0.58);

        // Test if S_{::+} = X
        tensord<2> sum_S = S.sum(shape<1>({2}));
        matrix_t sum_S_mat = Eigen::Map<matrix_t>(
            sum_S.data(), sum_S.dimension(0), sum_S.dimension(1));
        REQUIRE(X.isApprox(sum_S_mat));
    }
}

TEST_CASE("Parameter checks on collapsed gibbs", "[collapsed_gibbs]") {
    size_t x = 10, y = 5, z = 2;
    shape<3> tensor_shape{x, y, z};
    matrix_t X = matrix_t::Constant(x, y, 5);
    allocation_model::AllocModelParams model_params(tensor_shape);

    // max_iter = 5 since we don't want to wait
    REQUIRE_NOTHROW(bld::collapsed_gibbs(X, z, model_params, 5));

    X(0, 0) = -5;
    REQUIRE_THROWS(bld::collapsed_gibbs(X, z, model_params));

    X(0, 0) = 0;
    REQUIRE_NOTHROW(bld::collapsed_gibbs(X, z, model_params, 5));

    z++;
    REQUIRE_THROWS(bld::collapsed_gibbs(X, z, model_params));

    z--;
    X = matrix_t::Constant(x + 1, y, 5);
    REQUIRE_THROWS(bld::collapsed_gibbs(X, z, model_params));

    SECTION("Zero matrix throws exception") {
        X = matrix_t::Zero(x, y);
        REQUIRE_THROWS(bld::collapsed_gibbs(X, z, model_params));
    }
}

TEST_CASE("Algorithm checks on collapsed gibbs", "[collapsed_gibbs]") {
    size_t x = 8, y = 10, z = 3;
    shape<3> tensor_shape{x, y, z};

    SECTION("Test against the results of ppmf.py implementation on "
            "Experiments.ipynb") {
        matrix_t X(x, y);
        X << 6., 3., 6., 5., 6., 6., 7., 10., 12., 2., 1., 5., 5., 3., 1., 2.,
            8., 3., 2., 7., 3., 11., 5., 5., 2., 5., 5., 2., 0., 11., 1., 0.,
            4., 1., 1., 1., 1., 4., 1., 4., 8., 2., 3., 12., 10., 15., 6., 8.,
            18., 2., 10., 10., 5., 7., 8., 9., 9., 9., 24., 7., 4., 0., 1., 1.,
            5., 4., 2., 2., 9., 2., 4., 1., 3., 2., 7., 5., 4., 7., 10., 0.;

        AllocModelParams model_params(40, 1, std::vector<double>(x, 1.0),
                                      std::vector<double>(z, 1.0));

        auto gen = bld::collapsed_gibbs(X, z, model_params);
        for (const auto& sample : gen)
            ;

        tensord<3> S = *gen.begin();

        double log_marginal = log_marginal_S(S, model_params);

        REQUIRE(log_marginal >= -390);
        REQUIRE(sparseness(S) >= 0.40);

        tensord<2> sum_S = S.sum(shape<1>({2}));
        matrix_t sum_S_mat = Eigen::Map<matrix_t>(
            sum_S.data(), sum_S.dimension(0), sum_S.dimension(1));
        REQUIRE(X.isApprox(sum_S_mat));
    }
}

TEST_CASE("Parameter checks on collapsed icm", "[collapsed_icm]") {
    size_t x = 10, y = 5, z = 2;
    shape<3> tensor_shape{x, y, z};
    matrix_t X = matrix_t::Constant(x, y, 5);
    allocation_model::AllocModelParams model_params(tensor_shape);

    // max_iter = 5 since we don't want to wait
    REQUIRE_NOTHROW(bld::collapsed_icm(X, z, model_params, 5));

    X(0, 0) = -5;
    REQUIRE_THROWS(bld::collapsed_icm(X, z, model_params));

    X(0, 0) = 0;
    REQUIRE_NOTHROW(bld::collapsed_icm(X, z, model_params, 5));

    z++;
    REQUIRE_THROWS(bld::collapsed_icm(X, z, model_params));

    z--;
    X = matrix_t::Constant(x + 1, y, 5);
    REQUIRE_THROWS(bld::collapsed_icm(X, z, model_params));

    SECTION("Zero matrix throws exception") {
        X = matrix_t::Zero(x, y);
        REQUIRE_THROWS(bld::collapsed_gibbs(X, z, model_params));
    }
}

TEST_CASE("Algorithm checks on collapsed icm", "[collapsed_icm]") {
    size_t x = 8, y = 10, z = 3;
    shape<3> tensor_shape{x, y, z};

    SECTION("Test against the results of ppmf.py implementation on "
            "Experiments.ipynb") {
        matrix_t X(x, y);
        X << 6., 3., 6., 5., 6., 6., 7., 10., 12., 2., 1., 5., 5., 3., 1., 2.,
            8., 3., 2., 7., 3., 11., 5., 5., 2., 5., 5., 2., 0., 11., 1., 0.,
            4., 1., 1., 1., 1., 4., 1., 4., 8., 2., 3., 12., 10., 15., 6., 8.,
            18., 2., 10., 10., 5., 7., 8., 9., 9., 9., 24., 7., 4., 0., 1., 1.,
            5., 4., 2., 2., 9., 2., 4., 1., 3., 2., 7., 5., 4., 7., 10., 0.;

        AllocModelParams model_params(40, 1, std::vector<double>(x, 1.0),
                                      std::vector<double>(z, 1.0));

        tensord<3> S = bld::collapsed_icm(X, z, model_params);

        double log_marginal = log_marginal_S(S, model_params);

        REQUIRE(log_marginal >= -350);
        REQUIRE(sparseness(S) >= 0.55);

        tensord<2> sum_S = S.sum(shape<1>({2}));
        matrix_t sum_S_mat = Eigen::Map<matrix_t>(
            sum_S.data(), sum_S.dimension(0), sum_S.dimension(1));
        REQUIRE(X.isApprox(sum_S_mat));
    }
}

TEST_CASE("Parameter checks on bld approximate", "[bld_appr]") {
    size_t x = 10, y = 5, z = 2;
    shape<3> tensor_shape{x, y, z};
    matrix_t X = matrix_t::Constant(x, y, 5);
    allocation_model::AllocModelParams model_params(tensor_shape);

    // max_iter = 5 since we don't want to wait
    REQUIRE_NOTHROW(bld::bld_appr(X, z, model_params, 5));

    X(0, 0) = -5;
    REQUIRE_THROWS(bld::bld_appr(X, z, model_params));

    X(0, 0) = 0;
    REQUIRE_NOTHROW(bld::bld_appr(X, z, model_params, 5));

    z++;
    REQUIRE_THROWS(bld::bld_appr(X, z, model_params));

    z--;
    X = matrix_t::Constant(x + 1, y, 5);
    REQUIRE_THROWS(bld::bld_appr(X, z, model_params));
}

TEST_CASE("Algorithm checks on bld approximate", "[bld_appr]") {
    size_t x = 8, y = 10, z = 3;
    shape<3> tensor_shape{x, y, z};
    SECTION("Zero matrix result is Zero") {
        matrix_t X = matrix_t::Zero(x, y);
        AllocModelParams model_params(tensor_shape);

        tensord<3> S;
        matrix_t nu, mu;
        std::tie(S, nu, mu) = bld::bld_appr(X, z, model_params);

        tensord<0> min = S.minimum();
        tensord<0> max = S.maximum();
        REQUIRE(min.coeff() == Approx(0));
        REQUIRE(max.coeff() == Approx(0));
    }

    SECTION("Test against the results of ppmf.py implementation on "
            "Experiments.ipynb") {
        matrix_t X(x, y);
        X << 2., 0., 2., 2., 0., 2., 1., 0., 1., 3., 5., 4., 0., 2., 2., 6., 2.,
            1., 4., 5., 1., 13., 6., 9., 5., 8., 6., 6., 11., 12., 6., 2., 1.,
            4., 9., 7., 6., 10., 4., 8., 5., 5., 4., 3., 1., 0., 4., 8., 3., 1.,
            0., 9., 2., 1., 2., 7., 5., 3., 8., 3., 9., 8., 4., 5., 11., 9., 4.,
            6., 5., 12., 2., 6., 3., 3., 3., 6., 5., 3., 6., 4.;

        AllocModelParams model_params(40, 1, std::vector<double>(x, 1.0),
                                      std::vector<double>(z, 1.0));

        tensord<3> S;
        matrix_t nu, mu;
        std::tie(S, nu, mu) = bld::bld_appr(X, z, model_params);

        double log_marginal = log_marginal_S(S, model_params);

        REQUIRE(log_marginal >= -280);
        REQUIRE(sparseness(S) >= 0.50);

        tensord<2> sum_S = S.sum(shape<1>({2}));
        matrix_t sum_S_mat = Eigen::Map<matrix_t>(
            sum_S.data(), sum_S.dimension(0), sum_S.dimension(1));
        REQUIRE(X.isApprox(sum_S_mat));
    }
}
