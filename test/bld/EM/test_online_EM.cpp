#include "../../catch2.hpp"
#include "alloc_model/alloc_model_funcs.hpp"
#include "alloc_model/alloc_model_params.hpp"
#include "bld/bld_algs.hpp"
#include "util/util.hpp"
#include <iostream>
#include <iomanip>

using namespace bnmf_algs;

TEST_CASE("Parameter checks on EM", "[EM]") {
    size_t x = 10, y = 5, z = 2;
    shape<3> tensor_shape{x, y, z};
    matrixd X = matrixd::Constant(x, y, 5);
    auto param_vec = alloc_model::make_EM_params<double>(tensor_shape);

    // max_iter = 5 since we don't want to wait
    REQUIRE_NOTHROW(bld::online_EM(X, param_vec, 5));

    X(0, 0) = -5;
    REQUIRE_THROWS(bld::online_EM(X, param_vec));

    X(0, 0) = 0;
    REQUIRE_NOTHROW(bld::online_EM(X, param_vec, 5));

    X = matrixd::Constant(x + 1, y, 5);
    REQUIRE_THROWS(bld::online_EM(X, param_vec));
}

TEST_CASE("Algorithm checks on EM", "[EM]") {
    constexpr size_t x = 10, y = 5, z = 3;
    const std::vector<double> alpha(x, 0.1);
    const std::vector<double> beta(y, 10);
    const double b = 2;
    const alloc_model::Params<double> params(1, b, alpha, beta);
    const std::vector<alloc_model::Params<double>> param_vec(z, params);

    SECTION("Custom case 1") {
        matrixd X(x, y);
        X << 1, 2, 4, 4, 2, 1, NAN, 3, 2, 8, 3, 5, 3, 2, NAN, 7, 8, 1, 6, 9, 6,
            2, 1, 3, 3, 6, 3, 5, 2, 3, 9, 8, 9, 5, NAN, 4, 7, 5, 2, 2, 3, 2, 5,
            7, 3, NAN, 5, 4, NAN, NAN;


        auto em_res = bld::online_EM(X, param_vec, 1000);

        matrixd X_expected(x, y);
        X_expected << 1, 2, 4, 4, 2, 1, 3, 3, 2, 8, 3, 5, 3, 2, 5, 7, 8, 1, 6,
            9, 6, 2, 1, 3, 3, 6, 3, 5, 2, 3, 9, 8, 9, 5, 15, 4, 7, 5, 2, 2, 3,
            2, 5, 7, 3, 1417, 5, 4, 1414, 1422;

//        std::cout << X_expected << std::endl;
//        std::cout << std::endl;
        matrixd W = em_res.logW.array().exp();
        std::cout << em_res.log_PS.transpose() << std::endl;
//        REQUIRE(em_res.X_full.isApprox(X_expected));
    }
}
