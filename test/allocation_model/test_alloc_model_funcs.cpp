#include <random>

#include "../catch2.hpp"
#include "alloc_model/alloc_model_funcs.hpp"
#include "util/util.hpp"
#include <bitset>
#include <iostream>

using namespace bnmf_algs;
using namespace bnmf_algs::util;
using namespace bnmf_algs::alloc_model;

TEST_CASE("Parameter checks for bnmf_priors", "[bnmf_priors]") {
    size_t x = 5, y = 3, z = 2;
    shape<3> tensor_shape{x, y, z};
    Params<double> model_params(tensor_shape);
    matrixd W, H;
    vectord L;

    SECTION("Illegal parameters") {
        REQUIRE_NOTHROW(bnmf_priors<double>(tensor_shape, model_params));

        tensor_shape[0]++;
        REQUIRE_THROWS(bnmf_priors<double>(tensor_shape, model_params));

        model_params.alpha.resize(tensor_shape[0]);
        REQUIRE_NOTHROW(bnmf_priors<double>(tensor_shape, model_params));

        model_params.beta.resize(z + 2);
        REQUIRE_THROWS(bnmf_priors<double>(tensor_shape, model_params));

        tensor_shape[2] += 2;
        tensor_shape[1] = 0;
        REQUIRE_NOTHROW(bnmf_priors<double>(tensor_shape, model_params));

        tensor_shape[0] = 2;
        tensor_shape[1] = 1;
        tensor_shape[2] = 0;
        REQUIRE_THROWS(bnmf_priors<double>(tensor_shape, model_params));
    }

    SECTION("Shapes containing 0s") {
        for (unsigned long long i = 0; i < 8; ++i) {
            std::bitset<3> bits(i);
            tensor_shape = shape<3>{static_cast<size_t>(bits[0]),
                                    static_cast<size_t>(bits[1]),
                                    static_cast<size_t>(bits[2])};
            model_params.alpha.resize(tensor_shape[0]);
            model_params.beta.resize(tensor_shape[2]);

            std::tie(W, H, L) = bnmf_priors<double>(tensor_shape, model_params);
            REQUIRE(W.rows() == tensor_shape[0]);
            REQUIRE(W.cols() == tensor_shape[2]);
            REQUIRE(H.rows() == tensor_shape[2]);
            REQUIRE(H.cols() == tensor_shape[1]);
            REQUIRE(L.cols() == tensor_shape[1]);
        }
    }

    SECTION("Returned matrices have correct shapes") {
        std::tie(W, H, L) = bnmf_priors<double>(tensor_shape, model_params);

        REQUIRE(W.rows() == tensor_shape[0]);
        REQUIRE(W.cols() == tensor_shape[2]);
        REQUIRE(H.rows() == tensor_shape[2]);
        REQUIRE(H.cols() == tensor_shape[1]);
        REQUIRE(L.cols() == tensor_shape[1]);
    }
}

TEST_CASE("Parameter checks for sample_S using prior matrices", "[sample_S]") {
    size_t x = 10, y = 8, z = 5;
    matrixd W = matrixd::Ones(x, z);
    matrixd H = matrixd::Ones(z, y);
    vectord L = vectord::Ones(y);

    REQUIRE_NOTHROW(sample_S(W, H, L));

    L = vectord::Ones(y + 1);
    REQUIRE_THROWS(sample_S(W, H, L));

    H = matrixd::Ones(z, y + 1);
    W = matrixd::Ones(x, z + 2);
    REQUIRE_THROWS(sample_S(W, H, L));

    H = matrixd::Ones(z + 2, y + 1);
    REQUIRE_NOTHROW(sample_S(W, H, L));
}

TEST_CASE("Algorithm checks for bnmf_priors using distribution parameters",
          "[bnmf_priors]") {
    // result matrices
    size_t x = 7, y = 4, z = 8;
    shape<3> tensor_shape{x, y, z};
    matrixd W, H;
    vectord L;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(5, 10);

    // parameters
    double a = 5, b = 4;
    std::vector<double> alpha(x);
    std::vector<double> beta(z);
    for (size_t i = 0; i < x; ++i) {
        alpha[i] = dis(gen);
    }
    for (size_t i = 0; i < z; ++i) {
        beta[i] = dis(gen);
    }
    Params<double> model_params(a, b, alpha, beta);

    std::tie(W, H, L) = bnmf_priors<double>(tensor_shape, model_params);

    SECTION("All entries of W, H, L are nonnegative") {
        REQUIRE((W.array() >= 0).all());
        REQUIRE((H.array() >= 0).all());
        REQUIRE((L.array() >= 0).all());
    }

    SECTION("Columns of W and H sum to 1") {
        auto W_col_sums = W.colwise().sum().eval();
        REQUIRE(W_col_sums.minCoeff() == Approx(1));
        REQUIRE(W_col_sums.maxCoeff() == Approx(1));

        auto H_col_sums = H.colwise().sum().eval();
        REQUIRE(H_col_sums.minCoeff() == Approx(1));
        REQUIRE(H_col_sums.maxCoeff() == Approx(1));
    }
}

TEST_CASE("Algorithm checks for sample_S using prior matrices", "[sample_S]") {
    size_t x = 10, y = 8, z = 5;

    SECTION("Zero matrices") {
        matrixd W = matrixd::Zero(x, z);
        matrixd H = matrixd::Zero(z, y);
        vectord L = vectord::Ones(y);

        tensord<3> S = sample_S(W, H, L);
        double sum = 0;
        for (size_t i = 0; i < x; ++i)
            for (size_t j = 0; j < y; ++j)
                for (size_t k = 0; k < z; ++k)
                    sum += S(i, j, k);

        REQUIRE(sum == Approx(0));
    }

    SECTION("Custom matrices") {
        double scale = 5;
        matrixd W =
            (matrixd::Random(x, z) + matrixd::Constant(x, z, scale)) * scale;
        matrixd H =
            (matrixd::Random(z, y) + matrixd::Constant(z, y, scale)) * scale;
        vectord L = (vectord::Random(y) + vectord::Constant(y, scale)) * scale;
        tensord<3> S = sample_S(W, H, L);

        // TODO: How to check if the result comes from Poisson with certain mean
        SECTION("Nonnegativity check") {
            bool nonnegative = true;
            for (size_t i = 0; i < x; ++i)
                for (size_t j = 0; j < y; ++j)
                    for (size_t k = 0; k < z; ++k)
                        if (S(i, j, k) < 0)
                            nonnegative = false;
            REQUIRE(nonnegative);
        }

        // TODO: This will be deprecated once the return value of sample_S will
        // be an integer tensor as it should be
        SECTION("Check if matrix is integer valued") {
            tensord<3> S_round = S.round();
            double sum = 0;
            for (size_t i = 0; i < x; ++i)
                for (size_t j = 0; j < y; ++j)
                    for (size_t k = 0; k < z; ++k)
                        sum += (std::abs(S(i, j, k) - S_round(i, j, k)));

            REQUIRE(sum == Approx(0));
        }
    }
}

TEST_CASE("Algorithm checks for getting a sample from distribution parameters",
          "[bnmf_priors] [sample_S]") {
    // TODO: Implement (First need to understand the constraints on S)
}

TEST_CASE("Parameter checks on log marginal of S", "[log_marginal_S]") {
    size_t x = 5, y = 4, z = 8;
    shape<3> tensor_shape{x, y, z};
    tensord<3> S(x, y, z);
    S.setConstant(1);
    Params<double> model_params(tensor_shape);

    REQUIRE_NOTHROW(log_marginal_S(S, model_params));

    model_params.alpha.resize(x + 1);
    REQUIRE_THROWS(log_marginal_S(S, model_params));

    model_params.alpha.resize(x);
    model_params.beta.resize(z + 1);
    REQUIRE_THROWS(log_marginal_S(S, model_params));

    model_params.alpha.resize(x + 1);
    REQUIRE_THROWS(log_marginal_S(S, model_params));
}

TEST_CASE("Algorithm checks on log marginal of S", "[log_marginal_S]") {
    size_t x = 8, y = 10, z = 2;
    shape<3> tensor_shape{x, y, z};

    // parameters
    double a = 40, b = 1;
    std::vector<double> alpha(x);
    std::vector<double> beta(z);
    for (size_t i = 0; i < x; ++i) {
        alpha[i] = 1;
    }
    for (size_t i = 0; i < z; ++i) {
        beta[i] = 1;
    }
    Params<double> model_params(a, b, alpha, beta);

    tensord<3> S =
        call(sample_S<double>, bnmf_priors<double>(tensor_shape, model_params));

    SECTION("Changing original parameters results in lower likelihoods") {
        double original_log_marginal = log_marginal_S(S, model_params);

        // change parameters
        model_params.a = 200;
        model_params.b = 10;
        double altered_log_marginal = log_marginal_S(S, model_params);

        // Original likelihood must be higher
        REQUIRE(original_log_marginal > altered_log_marginal);

        // back to the original parameters; change alpha beta
        model_params.a = 100, model_params.b = 1;
        for (size_t i = 0; i < x; ++i) {
            model_params.alpha[i] = 0.2;
        }
        for (size_t i = 0; i < z; ++i) {
            model_params.beta[i] = 5;
        }
        altered_log_marginal = log_marginal_S(S, model_params);
        // Original likelihood must be higher
        REQUIRE(original_log_marginal > altered_log_marginal);
    }
}
