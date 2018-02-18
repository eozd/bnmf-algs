#include "catch2.hpp"
#include "bnmf_algs.hpp"
#include <fstream>
#include <iostream>

using namespace Eigen;

TEST_CASE("Euclidean NMF constraint checks", "nmf_euclidean") {
    int m = 50, n = 10, r = 5;

    MatrixXd X = MatrixXd::Random(m, n) + MatrixXd::Ones(m, n);
    MatrixXd W, H;
    std::tie(W, H) = bnmf_algs::nmf_euclidean(X, r);

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

TEST_CASE("Euclidean NMF invalid parameters", "nmf_euclidean") {
    int m = 500, n = 400, r = 20;
    MatrixXd X = MatrixXd::Zero(m, n);

    SECTION("Negative X", "negative-matrix") {
        X(0, 2) = -0.002;
        REQUIRE_THROWS(bnmf_algs::nmf_euclidean(X, r));
    }

    SECTION("Invalid inner dimension", "r") {
        REQUIRE_THROWS(bnmf_algs::nmf_euclidean(X, 0));
        REQUIRE_THROWS(bnmf_algs::nmf_euclidean(X, -1));
    }

    SECTION("Invalid max iter", "max-iter") {
        REQUIRE_THROWS(bnmf_algs::nmf_euclidean(X, r, -1));
    }

    SECTION("Invalid epsilon", "epsilon") {
        REQUIRE_THROWS(bnmf_algs::nmf_euclidean(X, r, 102, -0.00000001));
    }
}

TEST_CASE("Euclidean NMF degenerate cases", "degenerate") {
    int m = 500, n = 400, r = 20;

    SECTION("X == 0", "zero-matrix") {

        MatrixXd X = MatrixXd::Zero(m, n);
        MatrixXd W, H;
        std::tie(W, H) = bnmf_algs::nmf_euclidean(X, r, 200);

        REQUIRE(W.isZero(0));
        REQUIRE(H.isZero(0));
    }
}
