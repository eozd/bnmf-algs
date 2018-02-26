#include "catch2.hpp"
#include "sampling.hpp"
#include "util.hpp"

using namespace bnmf_algs;
using namespace bnmf_algs::util;

TEST_CASE("Test AllocModelParams construction", "[AllocModelParams]") {
    shape<3> tensor_shape{5, 2, 8};
    SECTION("Manual constructor") {
        double a = 5, b = 7;
        std::vector<double> alpha(tensor_shape[0], 5242);
        std::vector<double> beta(tensor_shape[2], 24);
        AllocModelParams model_params(a, b, alpha, beta);

        REQUIRE(model_params.a == a);
        REQUIRE(model_params.b == b);
        REQUIRE(model_params.alpha == alpha);
        REQUIRE(model_params.beta == beta);
    }

    SECTION("Default constructor") {
        AllocModelParams model_params(tensor_shape);

        double a_default = 1.0;
        double b_default = 10.0;
        std::vector<double> alpha_default(tensor_shape[0], 1.0);
        std::vector<double> beta_default(tensor_shape[2], 1.0);

        REQUIRE(model_params.a == a_default);
        REQUIRE(model_params.b == b_default);
        REQUIRE(model_params.alpha == alpha_default);
        REQUIRE(model_params.beta == beta_default);
    }
}
