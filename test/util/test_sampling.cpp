#include "../catch2.hpp"
#include "util/sampling.hpp"
#include <iostream>

using namespace bnmf_algs;

TEST_CASE("Parameter checks on sample ones no replace",
          "[sample_ones_noreplace]") {

    SECTION("Empty matrix X throws exception") {
        matrixd X;
        REQUIRE_THROWS(util::sample_ones_noreplace(X));

        X = matrixd::Zero(1, 0);
        REQUIRE_THROWS(util::sample_ones_noreplace(X));

        X = matrixd::Zero(0, 1);
        REQUIRE_THROWS(util::sample_ones_noreplace(X));

        X = matrixd::Ones(1, 1);
        REQUIRE_NOTHROW(util::sample_ones_noreplace(X));
    }

    SECTION("Matrix X with negative entries throws exception") {
        matrixd X = matrixd::Zero(5, 5);
        X(2, 3) = -1e-50;
        REQUIRE_THROWS(util::sample_ones_noreplace(X));
    }

    SECTION("Zero matrix throws exception") {
        matrixd X = matrixd::Zero(5, 5);
        REQUIRE_THROWS(util::sample_ones_noreplace(X));
    }

    SECTION("Number of samples equals to sum(X) when replacement is false") {
        size_t x = 10, y = 15;
        matrixd X = matrixd::Random(x, y) + matrixd::Constant(x, y, 1);
        auto gen = util::sample_ones_noreplace(X);

        std::vector<std::pair<int, int>> res;
        std::copy(gen.begin(), gen.end(), std::back_inserter(res));

        REQUIRE(res.size() == static_cast<size_t>(X.sum()));
    }
}

TEST_CASE("Algorithm checks on sample_ones no replace",
          "[sample_ones_noreplace]") {

    SECTION("First k 0-elements are never sampled") {
        size_t x = 10, y = 15;
        size_t num_zero_rows = 2;
        matrixd X = matrixd::Random(x, y) + matrixd::Constant(x, y, 5);
        X.block(0, 0, num_zero_rows, y) = matrixd::Zero(num_zero_rows, y);
        auto gen_no_replacement = util::sample_ones_noreplace(X);

        bool contains_invalid_sample =
            std::any_of(gen_no_replacement.begin(), gen_no_replacement.end(),
                        [](const auto& pair) { return pair.first <= 1; });
        REQUIRE(!contains_invalid_sample);
    }
}

TEST_CASE("Parameter checks on sample ones replace", "[sample_ones_replace]") {

    SECTION("Empty matrix X throws exception") {
        matrixd X;
        REQUIRE_THROWS(util::sample_ones_replace(X, 10));

        X = matrixd::Zero(1, 0);
        REQUIRE_THROWS(util::sample_ones_replace(X, 10));

        X = matrixd::Zero(0, 1);
        REQUIRE_THROWS(util::sample_ones_replace(X, 10));

        X = matrixd::Ones(1, 1);
        REQUIRE_NOTHROW(util::sample_ones_replace(X, 10));
    }

    SECTION("Matrix X with negative entries throws exception") {
        matrixd X = matrixd::Zero(5, 5);
        X(2, 3) = -1e-50;
        REQUIRE_THROWS(util::sample_ones_replace(X, 10));
    }

    SECTION("Zero matrix throws exception") {
        matrixd X = matrixd::Zero(5, 5);
        REQUIRE_THROWS(util::sample_ones_replace(X, 10));
    }

    SECTION("Number of samples equals to n when replacement is true") {
        size_t n = 491, x = 5, y = 3;
        matrixd X = matrixd::Random(x, y) + matrixd::Constant(x, y, 1);
        auto gen = util::sample_ones_replace(X, n);

        std::vector<std::pair<int, int>> res;
        std::copy(gen.begin(), gen.end(), std::back_inserter(res));

        REQUIRE(res.size() == n);

        n = 0;
        res.clear();
        gen = util::sample_ones_replace(X, n);
        std::copy(gen.begin(), gen.end(), std::back_inserter(res));

        REQUIRE(res.empty());
    }
}

TEST_CASE("Algorithm checks on sample_ones replace", "[sample_ones_replace]") {

    SECTION("First k 0-elements are never sampled") {
        size_t x = 10, y = 15;
        size_t num_zero_rows = 2;
        matrixd X = matrixd::Random(x, y) + matrixd::Constant(x, y, 5);
        X.block(0, 0, num_zero_rows, y) = matrixd::Zero(num_zero_rows, y);

        size_t num_samples = 1000;
        auto gen_replacement = util::sample_ones_replace(X, num_samples);

        bool contains_invalid_sample =
            std::any_of(gen_replacement.begin(), gen_replacement.end(),
                        [](const auto& pair) { return pair.first <= 1; });
        REQUIRE(!contains_invalid_sample);
    }
}

TEST_CASE("Test util::choice", "[choice]") {
    util::gsl_rng_wrapper rnd_gen(gsl_rng_alloc(gsl_rng_taus), gsl_rng_free);

    SECTION("Empty range") {
        std::vector<double> cum_prob;
        using iter = std::vector<double>::iterator;

        // choose 100 times
        std::vector<iter> chosen_iters;
        for (size_t i = 0; i < 100; ++i) {
            chosen_iters.push_back(
                util::choice(cum_prob.begin(), cum_prob.end(), rnd_gen));
        }
        // all chosen iterators must point to end
        REQUIRE(std::all_of(
            chosen_iters.begin(), chosen_iters.end(),
            [&cum_prob](const auto c_it) { return c_it == cum_prob.end(); }));
    }

    SECTION("Indicator function distribution") {
        size_t index = 4;
        std::vector<double> cum_prob(10, 0.0);
        for (size_t i = index; i < cum_prob.size(); ++i) {
            cum_prob[i] = 1;
        }

        // choose 100 times
        std::vector<size_t> chosen_indices;
        for (size_t i = 0; i < 100; ++i) {
            auto it = util::choice(cum_prob.begin(), cum_prob.end(), rnd_gen);
            chosen_indices.push_back(it - cum_prob.begin());
        }
        // all chosen indices must be the same
        REQUIRE(std::all_of(chosen_indices.begin(), chosen_indices.end(),
                            [index](const size_t ci) { return ci == index; }));
    }

    SECTION("All zeros distribution") {
        std::vector<double> cum_prob(10, 0.0);
        using iter = std::vector<double>::iterator;

        // choose 100 times
        std::vector<iter> chosen_iters;
        for (size_t i = 0; i < 100; ++i) {
            chosen_iters.push_back(
                util::choice(cum_prob.begin(), cum_prob.end(), rnd_gen));
        }
        // all returned iterators point to the end
        REQUIRE(std::all_of(
            chosen_iters.begin(), chosen_iters.end(),
            [&cum_prob](const auto c_it) { return c_it == cum_prob.end(); }));
    }
}

TEST_CASE("Test util::multinomial_mode", "[multinomial_mode]") {
    SECTION("Parameter checks") {
        const size_t num_events = 5;
        int num_balls = 0;
        vector_t<double> prob(num_events);
        prob << 0.2, 0.2, 0.2, 0.2, 0.2;

        vector_t<int> count(num_events);
        REQUIRE_NOTHROW(util::multinomial_mode(num_balls, prob, count));

        num_balls = -1;
        REQUIRE_THROWS(util::multinomial_mode(num_balls, prob, count));

        num_balls = 120;
        count = vector_t<int>(num_events - 1);
        REQUIRE_THROWS(util::multinomial_mode(num_balls, prob, count));
    }

    SECTION("Tests given in Finucan paper") {
        const size_t num_events = 6;
        const size_t num_balls = 17;
        vector_t<size_t> count(num_events);
        vector_t<size_t> expected(num_events);

        vector_t<double> prob(num_events);
        prob << 7, 12, 13, 19, 21, 28;
        prob = prob.array() / prob.sum();
        expected << 1, 2, 2, 3, 4, 5;
        util::multinomial_mode(num_balls, prob, count);
        REQUIRE(count == expected);

        prob << 6, 11, 12, 20, 23, 28;
        prob = prob.array() / prob.sum();
        expected << 1, 2, 2, 3, 4, 5;
        util::multinomial_mode(num_balls, prob, count);
        REQUIRE(count == expected);

        prob << 7, 11, 12, 21, 22, 27;
        prob = prob.array() / prob.sum();
        expected << 1, 2, 2, 3, 4, 5;
        util::multinomial_mode(num_balls, prob, count);
        REQUIRE(count == expected);

        prob << 6, 10.5, 11, 21, 21, 30.5;
        prob = prob.array() / prob.sum();
        expected << 1, 2, 2, 4, 3, 5; // one of the modes
        util::multinomial_mode(num_balls, prob, count);
        REQUIRE(count == expected);

        prob << 52, 104, 104, 208, 208, 324;
        prob = prob.array() / prob.sum();
        expected << 0, 2, 1, 4, 4, 6; // one of the modes
        util::multinomial_mode(num_balls, prob, count);
        REQUIRE(count == expected);

        prob << 13, 23, 25, 37, 41, 55;
        prob = prob.array() / prob.sum();
        expected << 1, 2, 2, 3, 4, 5;
        util::multinomial_mode(num_balls, prob, count);
        REQUIRE(count == expected);
    }

    SECTION("Edge case tests: 0 events") {
        const size_t num_events = 0;
        const size_t num_trials = 10;
        vector_t<double> prob;
        vector_t<size_t> count;
        util::multinomial_mode(num_trials, prob, count);
        REQUIRE(count == vector_t<size_t>());
    }

    SECTION("Edge case tests: 0 trials") {
        const size_t num_events = 3;
        const size_t num_trials = 0;
        vector_t<double> prob(num_events);
        prob << 1. / 3, 1. / 3, 1. / 3;
        vector_t<size_t> count(num_events);
        util::multinomial_mode(num_trials, prob, count);

        vector_t<size_t> expected(num_events);
        expected << 0, 0, 0;
        REQUIRE(count == expected);
    }

    SECTION("Custom test: 1 ball") {
        const size_t num_events = 4;
        const size_t num_trials = 1;
        vector_t<double> prob(num_events);
        prob << 0.75, 0.1, 0.1, 0.05;
        vector_t<size_t> count(num_events);
        util::multinomial_mode(num_trials, prob, count);

        vector_t<size_t> expected(num_events);
        expected << 1, 0, 0, 0;
        REQUIRE(count == expected);
    }

    SECTION("Custom test: 2 balls") {
        const size_t num_events = 4;
        const size_t num_trials = 2;
        vector_t<double> prob(num_events);
        prob << 0.4, 0.4, 0.2, 0;
        vector_t<size_t> count(num_events);
        util::multinomial_mode(num_trials, prob, count);

        vector_t<size_t> expected(num_events);
        expected << 1, 1, 0, 0;
        REQUIRE(count == expected);
    }

    SECTION("Custom test: Uniform distribution") {
        const size_t num_events = 4;
        const size_t num_trials = 60;
        vector_t<double> prob(num_events);
        prob << 0.25, 0.25, 0.25, 0.25;
        vector_t<size_t> count(num_events);
        util::multinomial_mode(num_trials, prob, count);

        vector_t<size_t> expected(num_events);
        expected << 15, 15, 15, 15;
        REQUIRE(count == expected);
    }

    SECTION("Custom test: dirac delta") {
        const size_t num_events = 5;
        const size_t num_trials = 60;
        vector_t<double> prob(num_events);
        prob << 0, 0, 1, 0, 0;
        vector_t<size_t> count(num_events);
        util::multinomial_mode(num_trials, prob, count);

        vector_t<size_t> expected(num_events);
        expected << 0, 0, 60, 0, 0;
        REQUIRE(count == expected);
    }
}

TEST_CASE("Test dirichlet_mat", "[dirichlet_mat]") {
    const size_t x = 50, y = 30;
    matrix_t<double> alpha =
        matrix_t<double>::Random(x, y) + matrix_t<double>::Constant(x, y, 1);

    SECTION("Parameter checks") {
        REQUIRE_NOTHROW(util::dirichlet_mat(alpha, 0));
        REQUIRE_NOTHROW(util::dirichlet_mat(alpha, 1));
        REQUIRE_THROWS(util::dirichlet_mat(alpha, 2));
    }

    SECTION("Columnwise normalized Dirichlet matrix") {
        matrix_t<double> dirichlet = util::dirichlet_mat(alpha, 0);
        vector_t<double> col_sums = dirichlet.colwise().sum();
        vector_t<double> expected = vector_t<double>::Constant(y, 1);

        REQUIRE(col_sums.isApprox(expected));
    }

    SECTION("Rowwise normalized Dirichlet matrix") {
        matrix_t<double> dirichlet = util::dirichlet_mat(alpha, 1);
        vector_t<double> row_sums = dirichlet.rowwise().sum();
        vector_t<double> expected = vector_t<double>::Constant(x, 1);

        REQUIRE(row_sums.isApprox(expected));
    }
}