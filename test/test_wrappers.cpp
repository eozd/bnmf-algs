#include "catch2.hpp"
#include "wrappers.hpp"

using namespace bnmf_algs;

TEST_CASE("Correct wrapper creation checks", "gsl_rng_wrapper") {
    SECTION("Check if we get a nullptr after normal construction", "nullptr") {
        auto rand_gen = make_gsl_rng(gsl_rng_taus);
        REQUIRE(rand_gen.get() != nullptr);
    }

    SECTION("Check if we can use the generator after normal construction", "rng") {
        auto rand_gen = make_gsl_rng(gsl_rng_uni);
        int a = 5, b = 10;
        double rand_num = gsl_ran_flat(rand_gen.get(), a, b);
        REQUIRE(rand_num >= a);
        REQUIRE(rand_num <= b);
    }
}
