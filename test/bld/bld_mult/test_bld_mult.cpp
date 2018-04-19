#include "../../catch2.hpp"
#include "alloc_model/alloc_model_funcs.hpp"
#include "alloc_model/alloc_model_params.hpp"
#include "bld/bld_algs.hpp"
#include "util/util.hpp"
#include <iostream>

using namespace bnmf_algs;

TEST_CASE("Parameter checks on bld_mult", "[bld_mult]") {
    size_t x = 10, y = 5, z = 2;
    shape<3> tensor_shape{x, y, z};
    matrixd X = matrixd::Constant(x, y, 5);
    alloc_model::Params<double> model_params(tensor_shape);

    // max_iter = 5 since we don't want to wait
    REQUIRE_NOTHROW(bld::bld_mult(X, z, model_params, 5));

    X(0, 0) = -5;
    REQUIRE_THROWS(bld::bld_mult(X, z, model_params));

    X(0, 0) = 0;
    REQUIRE_NOTHROW(bld::bld_mult(X, z, model_params, 5));

    z++;
    REQUIRE_THROWS(bld::bld_mult(X, z, model_params));

    z--;
    X = matrixd::Constant(x + 1, y, 5);
    REQUIRE_THROWS(bld::bld_mult(X, z, model_params));
}

TEST_CASE("Algorithm checks on bld_mult", "[bld_mult]") {
    size_t x = 8, y = 10, z = 3;
    shape<3> tensor_shape{x, y, z};
    SECTION("Zero matrix result is Zero") {
        matrixd X = matrixd::Zero(x, y);
        alloc_model::Params<double> model_params(tensor_shape);

        tensord<3> S = bld::bld_mult(X, z, model_params);

        tensord<0> min = S.minimum();
        tensord<0> max = S.maximum();
        REQUIRE(min.coeff() == Approx(0));
        REQUIRE(max.coeff() == Approx(0));
    }

    SECTION("Test against the results of ppmf.py implementation on "
            "Experiments.ipynb") {
        matrixd X(x, y);
        X << 7., 11., 5., 8., 8., 8., 7., 6., 3., 9., 3., 7., 3., 1., 0., 6.,
            1., 0., 1., 4., 1., 11., 4., 5., 3., 8., 5., 3., 3., 3., 0., 20.,
            11., 16., 14., 26., 16., 23., 16., 15., 0., 1., 0., 2., 2., 2., 0.,
            1., 0., 1., 3., 4., 1., 0., 4., 0., 2., 0., 1., 5., 4., 4., 2., 2.,
            3., 4., 0., 2., 0., 5., 4., 3., 3., 1., 2., 6., 5., 7., 4., 2.;

        alloc_model::Params<double> model_params(
            40, 1, std::vector<double>(x, 1.0), std::vector<double>(z, 1.0));

        // Test with exact digamma
        // =======================
        tensord<3> S = bld::bld_mult(X, z, model_params, 2000);
        double log_marginal = alloc_model::log_marginal_S(S, model_params);
        REQUIRE(log_marginal >= -262);
        REQUIRE(util::sparseness(S) >= 0.64);

        // Test if S_{::+} = X
        tensord<2> sum_S = S.sum(shape<1>({2}));
        matrixd sum_S_mat = Eigen::Map<matrixd>(
            sum_S.data(), sum_S.dimension(0), sum_S.dimension(1));
        REQUIRE(X.isApprox(sum_S_mat, 1e-10));

        // Test with approximate digamma
        // =============================
        S = bld::bld_mult(X, z, model_params, 2000, true);
        log_marginal = alloc_model::log_marginal_S(S, model_params);
        REQUIRE(log_marginal >= -262);
        REQUIRE(util::sparseness(S) >= 0.64);

        // Test if S_{::+} = X
        sum_S = S.sum(shape<1>({2}));
        sum_S_mat = Eigen::Map<matrixd>(sum_S.data(), sum_S.dimension(0),
                                        sum_S.dimension(1));
        REQUIRE(X.isApprox(sum_S_mat, 1e-4));
    }
}
