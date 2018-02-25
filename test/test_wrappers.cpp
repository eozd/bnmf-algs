#include <numeric>
#include "catch2.hpp"
#include "wrappers.hpp"

using namespace bnmf_algs;

TEST_CASE("gsl_rng_wrapper construction checks", "[gsl_rng_wrapper]") {
    SECTION("Check if we get a nullptr after normal construction") {
        auto rand_gen = make_gsl_rng(gsl_rng_taus);
        REQUIRE(rand_gen.get() != nullptr);
    }

    SECTION("Check if we can use the generator after normal construction") {
        auto rand_gen = make_gsl_rng(gsl_rng_uni);
        int a = 5, b = 10;
        double rand_num = gsl_ran_flat(rand_gen.get(), a, b);
        REQUIRE(rand_num >= a);
        REQUIRE(rand_num <= b);
    }
}

TEST_CASE("gsl_ran_discrete_wrapper construction checks",
          "[gsl_ran_discrete_wrapper]") {
    constexpr size_t K = 3;
    std::array<double, K> probs{5, 5, 5};

    SECTION("Uniform distribution") {
        auto ran_discrete = make_gsl_ran_discrete(K, probs.data());
        auto rand_gen = make_gsl_rng(gsl_rng_taus);
        std::vector<bool> hit(K, false);

        for (int i = 0; i < 500; ++i) {
            size_t index = gsl_ran_discrete(rand_gen.get(), ran_discrete.get());
            hit[index] = true;
        }
        // prob of not hitting one of the bins is 2.7e-88
        REQUIRE(std::accumulate(hit.begin(), hit.end(), 0) == K);
    }

    SECTION("Constant distribution") {
        probs = {0, 5, 0};
        auto ran_discrete = make_gsl_ran_discrete(K, probs.data());
        auto rand_gen = make_gsl_rng(gsl_rng_taus);
        std::vector<bool> hit(K, false);
        for (int i = 0; i < 500; ++i) {
            size_t index = gsl_ran_discrete(rand_gen.get(), ran_discrete.get());
            hit[index] = true;
        }

        REQUIRE(!hit[0]);
        REQUIRE(hit[1]);
        REQUIRE(!hit[2]);
    }
}