#include "../catch2.hpp"
#include "util/sampling.hpp"
#include <iostream>

using namespace bnmf_algs;


TEST_CASE("Parameter checks on sample ones", "[sample_ones]") {

    SECTION("Empty matrix X throws exception") {
        matrix_t X;
        REQUIRE_THROWS(util::sample_ones(X));

        X = matrix_t::Zero(1, 0);
        REQUIRE_THROWS(util::sample_ones(X));

        X = matrix_t::Zero(0, 1);
        REQUIRE_THROWS(util::sample_ones(X));

        X = matrix_t::Ones(1, 1);
        REQUIRE_NOTHROW(util::sample_ones(X));
    }

    SECTION("Matrix X with negative entries throws exception") {
        matrix_t X = matrix_t::Zero(5, 5);
        X(2, 3) = -1e-50;
        REQUIRE_THROWS(util::sample_ones(X));
    }

    SECTION("Zero matrix throws exception") {
        matrix_t X = matrix_t::Zero(5, 5);
        REQUIRE_THROWS(util::sample_ones(X));
    }

    SECTION("Number of samples equals to n when replacement is true") {
        size_t n = 491, x = 5, y = 3;
        matrix_t X = matrix_t::Random(x, y) + matrix_t::Constant(x, y, 1);
        auto gen = util::sample_ones(X, true, n);

        std::vector<std::pair<int, int>> res;
        std::copy(gen.begin(), gen.end(), std::back_inserter(res));

        REQUIRE(res.size() == n);

        n = 0;
        res.clear();
        gen = util::sample_ones(X, true, n);
        std::copy(gen.begin(), gen.end(), std::back_inserter(res));

        REQUIRE(res.empty());
    }

    SECTION("Number of samples equals to sum(X) when replacement is false") {
        size_t x = 10, y = 15;
        matrix_t X = matrix_t::Random(x, y) + matrix_t::Constant(x, y, 1);
        auto gen = util::sample_ones(X);

        std::vector<std::pair<int, int>> res;
        std::copy(gen.begin(), gen.end(), std::back_inserter(res));

        REQUIRE(res.size() == static_cast<size_t>(X.sum()));
    }
}

TEST_CASE("Algorithm checks on sample_ones", "[sample_ones]") {

    SECTION("First k 0-elements are never sampled") {
        size_t x = 10, y = 15;
        size_t num_zero_rows = 2;
        matrix_t X = matrix_t::Random(x, y) + matrix_t::Constant(x, y, 5);
        X.block(0, 0, num_zero_rows, y) = matrix_t::Zero(num_zero_rows, y);

        {
            size_t num_samples = 1000;
            auto gen_replacement = util::sample_ones(X, true, num_samples);

            bool contains_invalid_sample =
                    std::any_of(gen_replacement.begin(), gen_replacement.end(),
                                [](const auto& pair) { return pair.first <= 1; });
            REQUIRE(!contains_invalid_sample);
        }

        {
            auto gen_no_replacement = util::sample_ones(X);

            bool contains_invalid_sample = std::any_of(
                    gen_no_replacement.begin(), gen_no_replacement.end(),
                    [](const auto& pair) { return pair.first <= 1; });
            REQUIRE(!contains_invalid_sample);
        }
    }
}
