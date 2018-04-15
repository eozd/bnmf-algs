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

    matrixd X = matrixd::Random(m, n) + matrixd::Ones(m, n);
    matrixd W, H;
    std::tie(W, H) = nmf::nmf(X, r, 2);

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

    matrixd X = matrixd::Random(m, n) + matrixd::Ones(m, n);
    matrixd W, H;

    auto before = high_resolution_clock::now();
    std::tie(W, H) = nmf::nmf(X, r, 2, 500);
    auto after = high_resolution_clock::now();

    milliseconds elapsed = duration_cast<milliseconds>(after - before);
    REQUIRE(elapsed.count() <= 1000);
}

TEST_CASE("KL NMF small matrix convergence check", "[nmf]") {
    int m = 50, n = 10, r = 10;

    matrixd X = matrixd::Random(m, n) + matrixd::Ones(m, n);
    matrixd W, H;

    auto before = high_resolution_clock::now();
    std::tie(W, H) = nmf::nmf(X, r, 1, 500);
    auto after = high_resolution_clock::now();

    milliseconds elapsed = duration_cast<milliseconds>(after - before);
    REQUIRE(elapsed.count() <= 1000);
}

TEST_CASE("KL NMF constraint checks", "[nmf]") {
    int m = 10, n = 5, r = 2;

    matrixd X = matrixd::Random(m, n) + matrixd::Ones(m, n);
    matrixd W, H;
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
    matrixd X = matrixd::Zero(m, n);

    SECTION("Negative X") {
        X(0, 2) = -0.002;
        REQUIRE_THROWS(nmf::nmf(X, r, 2));
    }

    SECTION("Invalid inner dimension") { REQUIRE_THROWS(nmf::nmf(X, 0, 2)); }
}

TEST_CASE("Euclidean NMF degenerate cases", "[nmf]") {
    int m = 500, n = 400, r = 20;

    SECTION("X == 0") {

        matrixd X = matrixd::Zero(m, n);
        matrixd W, H;
        std::tie(W, H) = nmf::nmf(X, r, 2, 200);

        REQUIRE(W.isZero(0));
        REQUIRE(H.isZero(0));
    }
}

TEST_CASE("IS NMF small matrix convergence check", "[nmf]") {
    int m = 50, n = 10, r = 10;

    matrixd X = matrixd::Random(m, n) + matrixd::Ones(m, n);
    matrixd W, H;

    auto before = high_resolution_clock::now();
    std::tie(W, H) = nmf::nmf(X, r, 0, 500);
    auto after = high_resolution_clock::now();

    milliseconds elapsed = duration_cast<milliseconds>(after - before);
    REQUIRE(elapsed.count() <= 1000);
}

TEST_CASE("IS NMF constraint checks", "[nmf]") {
    int m = 10, n = 5, r = 2;

    matrixd X = matrixd::Random(m, n) + matrixd::Ones(m, n);
    matrixd W, H;
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

TEST_CASE("Test beta divergence for single values", "[beta-divergence]") {
    double x, y, beta;

    SECTION("x == 0 and y == 0") {
        x = 0, y = 0;

        beta = 0;
        REQUIRE(nmf::beta_divergence(x, y, beta) == Approx(-1));
        beta = 1;
        REQUIRE(nmf::beta_divergence(x, y, beta) == Approx(0));
        beta = 2;
        REQUIRE(nmf::beta_divergence(x, y, beta) == Approx(0));
    }

    SECTION("y == 1") {
        x = 0, y = 1;

        beta = 0;
        REQUIRE(nmf::beta_divergence(x, y, beta) == Approx(-1));
        beta = 1;
        REQUIRE(nmf::beta_divergence(x, y, beta) == Approx(1));
        beta = 2;
        REQUIRE(nmf::beta_divergence(x, y, beta) == Approx(0.5));

        x = 1;

        beta = 0;
        REQUIRE(nmf::beta_divergence(x, y, beta) == Approx(0));
        beta = 1;
        REQUIRE(nmf::beta_divergence(x, y, beta) == Approx(0));
        beta = 2;
        REQUIRE(nmf::beta_divergence(x, y, beta) == Approx(0));
    }

    SECTION("General case") {
        x = 2, y = 15;

        beta = 0;
        REQUIRE(nmf::beta_divergence(x, y, beta) == Approx(1.148236353875598));
        beta = 1;
        REQUIRE(nmf::beta_divergence(x, y, beta) == Approx(8.97019395891547));
        beta = 2;
        REQUIRE(nmf::beta_divergence(x, y, beta) == Approx(84.5));

        x = 15;

        beta = 0;
        REQUIRE(nmf::beta_divergence(x, y, beta) == Approx(0));
        beta = 1;
        REQUIRE(nmf::beta_divergence(x, y, beta) == Approx(0));
        beta = 2;
        REQUIRE(nmf::beta_divergence(x, y, beta) == Approx(0));
    }
}

TEST_CASE("Test beta divergence for sequences", "[beta-divergence]") {
    std::vector<double> x, y;
    double beta;

    SECTION("General case") {
        x = {1, 2, 3, 4, 5};
        y = {5, 4, 3, 2, 1};

        beta = 0;
        REQUIRE(nmf::beta_divergence(x.begin(), x.end(), y.begin(), beta) ==
                Approx(3.6999999999999993));

        beta = 1;
        REQUIRE(nmf::beta_divergence(x.begin(), x.end(), y.begin(), beta) ==
                Approx(7.824046010856292));

        beta = 2;
        REQUIRE(nmf::beta_divergence(x.begin(), x.end(), y.begin(), beta) ==
                Approx(20));
    }

    SECTION("Only x contains 0") {
        x = {1, 2, 0, 4, 5};
        y = {5, 4, 3, 2, 1};

        beta = 0;
        REQUIRE(nmf::beta_divergence(x.begin(), x.end(), y.begin(), beta) ==
                Approx(2.7));

        beta = 1;
        REQUIRE(nmf::beta_divergence(x.begin(), x.end(), y.begin(), beta) ==
                Approx(10.824046010856292));

        beta = 2;
        REQUIRE(nmf::beta_divergence(x.begin(), x.end(), y.begin(), beta) ==
                Approx(24.5));
    }


    SECTION("Only y contains 0") {
        double eps_32bit = std::numeric_limits<float>::epsilon();
        x = {1, 2, 3, 4, 5};
        y = {5, 4, 3, 0, 1};

        beta = 0;
        REQUIRE(Approx(33554417.06446767) ==
                nmf::beta_divergence(x.begin(), x.end(), y.begin(), beta,
                                     eps_32bit));

        beta = 1;
        REQUIRE(nmf::beta_divergence(x.begin(), x.end(), y.begin(), beta,
                                     eps_32bit) ==
                Approx(72.36617534461104));

        beta = 2;
        REQUIRE(nmf::beta_divergence(x.begin(), x.end(), y.begin(), beta) ==
                Approx(26));
    }

    SECTION("Overlapping 0s in x and y") {
        x = {1, 2, 0, 4, 5};
        y = {5, 4, 0, 2, 1};

        beta = 0;
        REQUIRE(nmf::beta_divergence(x.begin(), x.end(), y.begin(), beta) ==
                Approx(2.7));

        beta = 1;
        REQUIRE(nmf::beta_divergence(x.begin(), x.end(), y.begin(), beta) ==
                Approx(7.824046010856292));

        beta = 2;
        REQUIRE(nmf::beta_divergence(x.begin(), x.end(), y.begin(), beta) ==
                Approx(20));
    }
}

TEST_CASE("Test beta divergence for tensor-like objects", "[beta-divergence]") {
    vectord x(5), y(5);
    double beta;

    SECTION("General case") {
        x << 1, 2, 3, 4, 5;
        y << 5, 4, 3, 2, 1;

        beta = 0;
        REQUIRE(nmf::beta_divergence(x, y, beta) == Approx(3.6999999999999993));

        beta = 1;
        REQUIRE(nmf::beta_divergence(x, y, beta) == Approx(7.824046010856292));

        beta = 2;
        REQUIRE(nmf::beta_divergence(x, y, beta) == Approx(20));
    }

    SECTION("Only x contains 0") {
        x << 1, 2, 0, 4, 5;
        y << 5, 4, 3, 2, 1;

        beta = 0;
        REQUIRE(nmf::beta_divergence(x, y, beta) == Approx(2.7));

        beta = 1;
        REQUIRE(nmf::beta_divergence(x, y, beta) == Approx(10.824046010856292));

        beta = 2;
        REQUIRE(nmf::beta_divergence(x, y, beta) == Approx(24.5));
    }

    SECTION("Only y contains 0") {
        double eps_32bit = std::numeric_limits<float>::epsilon();
        x << 1, 2, 3, 4, 5;
        y << 5, 4, 3, 0, 1;

        beta = 0;
        REQUIRE(Approx(33554417.06446767) ==
                nmf::beta_divergence(x, y, beta, eps_32bit));

        beta = 1;
        REQUIRE(nmf::beta_divergence(x, y, beta, eps_32bit) ==
                Approx(72.36617534461104));

        beta = 2;
        REQUIRE(nmf::beta_divergence(x, y, beta) == Approx(26));
    }

    SECTION("Overlapping 0s in x and y") {
        x << 1, 2, 0, 4, 5;
        y << 5, 4, 0, 2, 1;

        beta = 0;
        REQUIRE(nmf::beta_divergence(x, y, beta) ==
                Approx(2.7));

        beta = 1;
        REQUIRE(nmf::beta_divergence(x, y, beta) ==
                Approx(7.824046010856292));

        beta = 2;
        REQUIRE(nmf::beta_divergence(x, y, beta) ==
                Approx(20));
    }
}
