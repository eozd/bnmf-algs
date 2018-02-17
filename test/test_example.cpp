#include "catch2.hpp"
#include "library.hpp"

TEST_CASE("Example test case", "[example]") {
    REQUIRE(bnmf_algs::hello(5) == 5);
}
