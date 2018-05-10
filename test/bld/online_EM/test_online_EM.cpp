#include "../../catch2.hpp"
#include "alloc_model/alloc_model_funcs.hpp"
#include "alloc_model/alloc_model_params.hpp"
#include "bld/bld_algs.hpp"
#include "util/util.hpp"
#include <iostream>

using namespace bnmf_algs;

TEST_CASE("Parameter checks on online_EM", "[online_EM]") {
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
