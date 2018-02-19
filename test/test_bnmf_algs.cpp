#include "catch2.hpp"
#include "bnmf_algs.hpp"
#include <fstream>
#include <iostream>

using namespace Eigen;
using namespace bnmf_algs;

TEST_CASE("Euclidean NMF constraint checks", "nmf") {
    int m = 50, n = 10, r = 5;

    MatrixXd X = MatrixXd::Random(m, n) + MatrixXd::Ones(m, n);
    MatrixXd W, H;
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

TEST_CASE("Euclidean NMF invalid parameters", "nmf") {
    int m = 500, n = 400, r = 20;
    MatrixXd X = MatrixXd::Zero(m, n);

    SECTION("Negative X", "negative-matrix") {
        X(0, 2) = -0.002;
        REQUIRE_THROWS(nmf(X, r, NMFVariant::Euclidean));
    }

    SECTION("Invalid inner dimension", "r") {
        REQUIRE_THROWS(nmf(X, 0, NMFVariant::Euclidean));
        REQUIRE_THROWS(nmf(X, -1, NMFVariant::Euclidean));
    }

    SECTION("Invalid max iter", "max-iter") {
        REQUIRE_THROWS(nmf(X, r, NMFVariant::Euclidean, -1));
    }

    SECTION("Invalid epsilon", "epsilon") {
        REQUIRE_THROWS(nmf(X, r, NMFVariant::Euclidean, 102, -0.00000001));
    }
}

TEST_CASE("Euclidean NMF degenerate cases", "degenerate") {
    int m = 500, n = 400, r = 20;

    SECTION("X == 0", "zero-matrix") {

        MatrixXd X = MatrixXd::Zero(m, n);
        MatrixXd W, H;
        std::tie(W, H) = nmf(X, r, NMFVariant::Euclidean, 200);

        REQUIRE(W.isZero(0));
        REQUIRE(H.isZero(0));
    }
}
