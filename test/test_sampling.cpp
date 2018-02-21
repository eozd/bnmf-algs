#include <random>

#include "sampling.hpp"
#include "catch2.hpp"


using namespace bnmf_algs;

TEST_CASE("Parameter checks for bnmf_priors", "bnmf_priors") {
    int x = 5, y = 3, z = 2;
    int a = 3, b = 2;
    shape<3> tensor_shape{x, y, z};
    std::vector<double> alpha(x);
    std::vector<double> beta(z);

    REQUIRE_NOTHROW(bnmf_priors(tensor_shape, a, b, alpha, beta));

    tensor_shape[0]++;
    REQUIRE_THROWS(bnmf_priors(tensor_shape, a, b, alpha, beta));

    alpha.resize(tensor_shape[0]);
    REQUIRE_NOTHROW(bnmf_priors(tensor_shape, a, b, alpha, beta));

    beta.resize(z + 2);
    REQUIRE_THROWS(bnmf_priors(tensor_shape, a, b, alpha, beta));

    tensor_shape[2] += 2;
    tensor_shape[1] = 0;
    REQUIRE_THROWS(bnmf_priors(tensor_shape, a, b, alpha, beta));

    tensor_shape[0] = -1;
    tensor_shape[1] = 1;
    REQUIRE_THROWS(bnmf_priors(tensor_shape, a, b, alpha, beta));

    tensor_shape[0] = 2;
    tensor_shape[1] = 1;
    tensor_shape[2] = 0;
    REQUIRE_THROWS(bnmf_priors(tensor_shape, a, b, alpha, beta));
}

TEST_CASE("Parameter checks for sample_S using prior matrices", "sample_S") {
    int x = 10, y = 8, z = 5;
    matrix_t W = matrix_t::Ones(x, z);
    matrix_t H = matrix_t::Ones(z, y);
    vector_t L = vector_t::Ones(y);

    REQUIRE_NOTHROW(sample_S(W, H, L));

    L = vector_t::Ones(y + 1);
    REQUIRE_THROWS(sample_S(W, H, L));

    H = matrix_t::Ones(z, y + 1);
    W = matrix_t::Ones(x, z + 2);
    REQUIRE_THROWS(sample_S(W, H, L));

    H = matrix_t::Ones(z + 2, y + 1);
    REQUIRE_NOTHROW(sample_S(W, H, L));
}

TEST_CASE("Algorithm checks for bnmf_priors using distribution parameters", "bnmf_priors") {
    // result matrices
    int x = 7, y = 4, z = 8;
    shape<3> tensor_shape{x, y, z};
    matrix_t W, H;
    vector_t L;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(5, 10);

    // parameters
    double a = 5, b = 4;
    std::vector<double> alpha(x);
    std::vector<double> beta(z);
    for (int i = 0; i < x; ++i) {
        alpha[i] = dis(gen);
    }
    for (int i = 0; i < z; ++i) {
        beta[i] = dis(gen);
    }

    std::tie(W, H, L) = bnmf_priors(tensor_shape, a, b, alpha, beta);

    SECTION("All entries of W, H, L are nonnegative", "nonnegative") {
        REQUIRE((W.array() >= 0).all());
        REQUIRE((H.array() >= 0).all());
        REQUIRE((L.array() >= 0).all());
    }

    SECTION("Columns of W and H sum to 1", "normalization") {
        auto W_col_sums = W.colwise().sum().eval();
        REQUIRE(W_col_sums.minCoeff() == Approx(1));
        REQUIRE(W_col_sums.maxCoeff() == Approx(1));

        auto H_col_sums = H.colwise().sum().eval();
        REQUIRE(H_col_sums.minCoeff() == Approx(1));
        REQUIRE(H_col_sums.maxCoeff() == Approx(1));
    }
}

TEST_CASE("Algorithm checks for sample_S using prior matrices", "sample_S") {
    int x = 10, y = 8, z = 5;

    SECTION("Zero matrices", "zero_matrices") {
        matrix_t W = matrix_t::Zero(x, z);
        matrix_t H = matrix_t::Zero(z, y);
        vector_t L = vector_t::Ones(y);

        tensor3d_t S = sample_S(W, H, L);
        double sum = 0;
        for (int i = 0; i < x; ++i)
            for (int j = 0; j < y; ++j)
                for (int k = 0; k < z; ++k)
                    sum += S(i, j, k);

        REQUIRE(sum == Approx(0));
    }

    SECTION("Custom matrices", "custom_matrices") {
        double scale = 5;
        matrix_t W = (matrix_t::Random(x, z) + matrix_t::Constant(x, z, scale))*scale;
        matrix_t H = (matrix_t::Random(z, y) + matrix_t::Constant(z, y, scale))*scale;
        vector_t L = (vector_t::Random(y) + vector_t::Constant(y, scale))*scale;
        tensor3d_t S = sample_S(W, H, L);

        // TODO: How to check if the result comes from Poisson with certain mean
        SECTION("Nonnegativity check", "nonnegativity") {
            bool nonnegative = true;
            for (int i = 0; i < x; ++i)
                for (int j = 0; j < y; ++j)
                    for (int k = 0; k < z; ++k)
                        if (S(i, j, k) < 0)
                            nonnegative = false;
            REQUIRE(nonnegative);
        }

        // TODO: This will be deprecated once the return value of sample_S will be an integer tensor as it should be
        SECTION("Check if matrix is integer valued", "integer") {
            tensor3d_t S_round = S.round();
            double sum = 0;
            for (int i = 0; i < x; ++i)
                for (int j = 0; j < y; ++j)
                    for (int k = 0; k < z; ++k)
                        sum += (std::abs(S(i, j, k) - S_round(i, j, k)));

            REQUIRE(sum == Approx(0));
        }
    }
}

TEST_CASE("Algorithm checks for getting a sample from distribution parameters", "bnmf_priors sample_S") {
    // TODO: Implement (First need to understand the constraints on S)
}
