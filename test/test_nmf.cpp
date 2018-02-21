#include "catch2.hpp"
#include "nmf.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <chrono>

using namespace std::chrono;
using namespace Eigen;
using namespace bnmf_algs;

TEST_CASE("Euclidean NMF constraint checks", "nmf") {
    int m = 50, n = 10, r = 5;

    matrix_t X = matrix_t::Random(m, n) + matrix_t::Ones(m, n);
    matrix_t W, H;
    std::tie(W, H) = nmf(X, r, NMFVariant::Euclidean);

    SECTION("Check returned matrices' shapes", "shape") {
        REQUIRE(W.rows() == m);
        REQUIRE(W.cols() == r);
        REQUIRE(H.rows() == r);
        REQUIRE(H.cols() == n);
    }

    SECTION("Check that returned matrices are nonnegative", "signs") {
        REQUIRE(W.minCoeff() >= 0);
        REQUIRE(H.minCoeff() >= 0);
    }
}

/**
 * @todo Need to check if errors are monotonically decreasing. For this, we may
 * need to refactor all NMF related functions, add some more information related
 * functionality and create a class or something similar.
 */
TEST_CASE("Euclidean NMF small matrix convergence check", "nmf") {
    int m = 50, n = 10, r = 10;

    matrix_t X = matrix_t::Random(m, n) + matrix_t::Ones(m, n);
    matrix_t W, H;

    auto before = high_resolution_clock::now();
    std::tie(W, H) = nmf(X, r, NMFVariant::Euclidean, 100000000, 1);
    auto after = high_resolution_clock::now();

    milliseconds elapsed = duration_cast<milliseconds>(after - before);
    REQUIRE(elapsed.count() <= 1000);
}

TEST_CASE("KL NMF small matrix convergence check", "nmf") {
    int m = 50, n = 10, r = 10;

    matrix_t X = matrix_t::Random(m, n) + matrix_t::Ones(m, n);
    matrix_t W, H;

    auto before = high_resolution_clock::now();
    std::tie(W, H) = nmf(X, r, NMFVariant::KL, 100000000, 1);
    auto after = high_resolution_clock::now();

    milliseconds elapsed = duration_cast<milliseconds>(after - before);
    REQUIRE(elapsed.count() <= 1000);
}

TEST_CASE("KL NMF constraint checks", "nmf") {
    int m = 50, n = 10, r = 5;

    matrix_t X = matrix_t::Random(m, n) + matrix_t::Ones(m, n);
    matrix_t W, H;
    std::tie(W, H) = nmf(X, r, NMFVariant::KL);

    SECTION("Check returned matrices' shapes", "shape") {
        REQUIRE(W.rows() == m);
        REQUIRE(W.cols() == r);
        REQUIRE(H.rows() == r);
        REQUIRE(H.cols() == n);
    }

    SECTION("Check that returned matrices are nonnegative", "signs") {
        REQUIRE(W.minCoeff() >= 0);
        REQUIRE(H.minCoeff() >= 0);
    }
}

TEST_CASE("Euclidean NMF invalid parameters", "nmf") {
    int m = 500, n = 400, r = 20;
    matrix_t X = matrix_t::Zero(m, n);

    SECTION("Negative X", "negative_matrix") {
        X(0, 2) = -0.002;
        REQUIRE_THROWS(nmf(X, r, NMFVariant::Euclidean));
    }

    SECTION("Invalid inner dimension", "r") {
        REQUIRE_THROWS(nmf(X, 0, NMFVariant::Euclidean));
        REQUIRE_THROWS(nmf(X, -1, NMFVariant::Euclidean));
    }

    SECTION("Invalid max iter", "max_iter") {
        REQUIRE_THROWS(nmf(X, r, NMFVariant::Euclidean, -1));
    }

    SECTION("Invalid epsilon", "epsilon") {
        REQUIRE_THROWS(nmf(X, r, NMFVariant::Euclidean, 102, -0.00000001));
    }
}

TEST_CASE("Euclidean NMF degenerate cases", "degenerate") {
    int m = 500, n = 400, r = 20;

    SECTION("X == 0", "zero-matrix") {

        matrix_t X = matrix_t::Zero(m, n);
        matrix_t W, H;
        std::tie(W, H) = nmf(X, r, NMFVariant::Euclidean, 200);

        REQUIRE(W.isZero(0));
        REQUIRE(H.isZero(0));
    }
}

TEST_CASE("Euclidean distance tests", "euclidean_cost") {
    int m = 100, n = 40;
    matrix_t X = matrix_t::Random(m, n);

    SECTION("Same matrices have 0 distance", "same_matrices") {
        REQUIRE(bnmf_algs::euclidean_cost(X, X) == Approx(0.0));
    }

    SECTION("One of the matrices is 0", "zero_matrix") {
        matrix_t Y = matrix_t::Zero(m, n);
        double x_norm_squared = X.norm()*X.norm();
        REQUIRE(bnmf_algs::euclidean_cost(X, Y) == Approx(x_norm_squared));
        REQUIRE(bnmf_algs::euclidean_cost(Y, X) == Approx(x_norm_squared));
    }

    SECTION("Two random matrices", "random_matrices") {
        matrix_t Y = matrix_t::Random(m, n);
        double norm = (X - Y).norm();
        double norm_squared = norm*norm;
        REQUIRE(bnmf_algs::euclidean_cost(X, Y) == Approx(norm_squared));
        REQUIRE(bnmf_algs::euclidean_cost(Y, X) == Approx(norm_squared));
    }
}

TEST_CASE("KL-divergence cost tests", "kl_cost") {
    int m = 100, n = 40;
    matrix_t X = matrix_t::Random(m, n);

    SECTION("Same matrices have 0 divergence", "same_matrices") {
        REQUIRE(bnmf_algs::kl_cost(X, X) == Approx(0.0));
    }

    SECTION("At least one of the matrices is 0", "zero_matrix") {
        matrix_t Y = X;
        X = matrix_t::Zero(m, n);
        REQUIRE(!std::isnan(bnmf_algs::kl_cost(X, Y)));
        REQUIRE(!std::isnan(bnmf_algs::kl_cost(Y, X)));

        Y = X;
        REQUIRE(!std::isnan(bnmf_algs::kl_cost(Y, X)));
    }

    SECTION("General correctness", "correctness") {
        matrix_t Z(3, 2);
        Z << 0.16012694, 0.71662857, 0.11937736,
                0.43474739, 0.7392143, 0.86325228;

        matrix_t Y(3, 2);
        Y << 0.75314334, 0.7584056, 0.54661004,
                0.23851888, 0.75112086, 0.2180139;

        REQUIRE(bnmf_algs::kl_cost(Z, Y) == Approx(1.19944962));
    }
}