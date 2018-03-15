#include "../catch2.hpp"
#include "nmf/nmf.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

using namespace std::chrono;
using namespace Eigen;
using namespace bnmf_algs;

TEST_CASE("Euclidean NMF constraint checks", "[nmf]") {
    int m = 10, n = 5, r = 2;

    matrix_t X = matrix_t::Random(m, n) + matrix_t::Ones(m, n);
    matrix_t W, H;
    std::tie(W, H) = nmf::nmf(X, r, 0);

    SECTION("Check returned matrices' shapes") {
        REQUIRE(W.rows() == m);
        REQUIRE(W.cols() == r);
        REQUIRE(H.rows() == r);
        REQUIRE(H.cols() == n);
    }

    SECTION("Check that returned matrices are nonnegative") {
        REQUIRE(W.minCoeff() >= 0);
        REQUIRE(H.minCoeff() >= 0);
    }
}

/**
 * @todo Need to check if errors are monotonically decreasing. For this, we may
 * need to refactor all NMF related functions, add some more information related
 * functionality and create a class or something similar.
 */
TEST_CASE("Euclidean NMF small matrix convergence check", "[nmf]") {
    int m = 50, n = 10, r = 10;

    matrix_t X = matrix_t::Random(m, n) + matrix_t::Ones(m, n);
    matrix_t W, H;

    auto before = high_resolution_clock::now();
    std::tie(W, H) = nmf::nmf(X, r, 2);
    auto after = high_resolution_clock::now();

    milliseconds elapsed = duration_cast<milliseconds>(after - before);
    REQUIRE(elapsed.count() <= 1000);
}

TEST_CASE("KL NMF small matrix convergence check", "[nmf]") {
    int m = 50, n = 10, r = 10;

    matrix_t X = matrix_t::Random(m, n) + matrix_t::Ones(m, n);
    matrix_t W, H;

    auto before = high_resolution_clock::now();
    std::tie(W, H) = nmf::nmf(X, r, 1, 1000);
    auto after = high_resolution_clock::now();

    milliseconds elapsed = duration_cast<milliseconds>(after - before);
    REQUIRE(elapsed.count() <= 1000);
}

TEST_CASE("KL NMF constraint checks", "[nmf]") {
    int m = 10, n = 5, r = 2;

    matrix_t X = matrix_t::Random(m, n) + matrix_t::Ones(m, n);
    matrix_t W, H;
    std::tie(W, H) = nmf::nmf(X, r, 1);

    SECTION("Check returned matrices' shapes") {
        REQUIRE(W.rows() == m);
        REQUIRE(W.cols() == r);
        REQUIRE(H.rows() == r);
        REQUIRE(H.cols() == n);
    }

    SECTION("Check that returned matrices are nonnegative") {
        REQUIRE(W.minCoeff() >= 0);
        REQUIRE(H.minCoeff() >= 0);
    }
}

TEST_CASE("Euclidean NMF invalid parameters", "[nmf]") {
    int m = 500, n = 400, r = 20;
    matrix_t X = matrix_t::Zero(m, n);

    SECTION("Negative X") {
        X(0, 2) = -0.002;
        REQUIRE_THROWS(nmf::nmf(X, r, 2));
    }

    SECTION("Invalid inner dimension") {
        REQUIRE_THROWS(nmf::nmf(X, 0, 2));
    }

}

TEST_CASE("Euclidean NMF degenerate cases", "[degenerate]") {
    int m = 500, n = 400, r = 20;

    SECTION("X == 0") {

        matrix_t X = matrix_t::Zero(m, n);
        matrix_t W, H;
        std::tie(W, H) = nmf::nmf(X, r, 2, 200);

        REQUIRE(W.isZero(0));
        REQUIRE(H.isZero(0));
    }
}
