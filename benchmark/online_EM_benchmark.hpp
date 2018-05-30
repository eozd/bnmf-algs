#pragma once

/**
 * This is the benchmark file to test bnmf_algs::bld::online_EM algorithm
 * performance when matrix size is increased.
 */

#include <celero/Celero.h>

#include "bench_utils.hpp"
#include "bnmf_algs.hpp"
#include <fstream>
#include <iostream>

namespace online_EM_bench_vars {
    size_t x = 10;
    size_t y = 100;
    double beg = 0;
    double scale = 5;
    size_t r = 17;
} // namespace online_EM_bench_vars

/**************************** Original bld_mult *******************************/

BASELINE(online_EM, 100x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace online_EM_bench_vars;

    x = 100;
    y = 100;

    shape<3> tensor_shape{x, y, r};
    matrixd X = make_matrix(x, y, beg, scale);
    auto param_vec = alloc_model::make_EM_params<double>(tensor_shape);
    celero::DoNotOptimizeAway(bld::online_EM(X, param_vec, 1000));
}

BENCHMARK(online_EM, 200x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace online_EM_bench_vars;

    x = 200;
    y = 100;

    shape<3> tensor_shape{x, y, r};
    matrixd X = make_matrix(x, y, beg, scale);
    auto param_vec = alloc_model::make_EM_params<double>(tensor_shape);
    celero::DoNotOptimizeAway(bld::online_EM(X, param_vec, 1000));
}

BENCHMARK(online_EM, 300x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace online_EM_bench_vars;

    x = 300;
    y = 100;

    shape<3> tensor_shape{x, y, r};
    matrixd X = make_matrix(x, y, beg, scale);
    auto param_vec = alloc_model::make_EM_params<double>(tensor_shape);
    celero::DoNotOptimizeAway(bld::online_EM(X, param_vec, 1000));
}

BENCHMARK(online_EM, 400x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace online_EM_bench_vars;

    x = 400;
    y = 100;

    shape<3> tensor_shape{x, y, r};
    matrixd X = make_matrix(x, y, beg, scale);
    auto param_vec = alloc_model::make_EM_params<double>(tensor_shape);
    celero::DoNotOptimizeAway(bld::online_EM(X, param_vec, 1000));
}

BENCHMARK(online_EM, 100x200, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace online_EM_bench_vars;

    x = 100;
    y = 200;

    shape<3> tensor_shape{x, y, r};
    matrixd X = make_matrix(x, y, beg, scale);
    auto param_vec = alloc_model::make_EM_params<double>(tensor_shape);
    celero::DoNotOptimizeAway(bld::online_EM(X, param_vec, 1000));
}

BENCHMARK(online_EM, 100x300, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace online_EM_bench_vars;

    x = 100;
    y = 300;

    shape<3> tensor_shape{x, y, r};
    matrixd X = make_matrix(x, y, beg, scale);
    auto param_vec = alloc_model::make_EM_params<double>(tensor_shape);
    celero::DoNotOptimizeAway(bld::online_EM(X, param_vec, 1000));
}

BENCHMARK(online_EM, 100x400, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace online_EM_bench_vars;

    x = 100;
    y = 400;

    shape<3> tensor_shape{x, y, r};
    matrixd X = make_matrix(x, y, beg, scale);
    auto param_vec = alloc_model::make_EM_params<double>(tensor_shape);
    celero::DoNotOptimizeAway(bld::online_EM(X, param_vec, 1000));
}

