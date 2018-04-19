#pragma once

/**
 * This is the benchmark file to test bnmf_algs::bld::collapsed_icm algorithm
 * performance when matrix size is increased.
 */

#include <celero/Celero.h>

#include "bench_utils.hpp"
#include "bnmf_algs.hpp"
#include <fstream>
#include <iostream>

namespace collapsed_icm_bench_vars {
long x = 100;
long y = 100;
double beg = 0;
double scale = 5;
size_t r = 17;
} // namespace collapsed_icm_bench_vars

BASELINE(collapsed_icm, 100x100, 30, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace collapsed_icm_bench_vars;

    x = 100;
    y = 100;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params<double>(x, y, r);
    celero::DoNotOptimizeAway(bld::collapsed_icm(X, r, params));
}

BENCHMARK(collapsed_icm, 200x100, 30, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace collapsed_icm_bench_vars;

    x = 200;
    y = 100;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params<double>(x, y, r);
    celero::DoNotOptimizeAway(bld::collapsed_icm(X, r, params));
}

BENCHMARK(collapsed_icm, 300x100, 30, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace collapsed_icm_bench_vars;

    x = 300;
    y = 100;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params<double>(x, y, r);
    celero::DoNotOptimizeAway(bld::collapsed_icm(X, r, params));
}

BENCHMARK(collapsed_icm, 400x100, 30, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace collapsed_icm_bench_vars;

    x = 400;
    y = 100;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params<double>(x, y, r);
    celero::DoNotOptimizeAway(bld::collapsed_icm(X, r, params));
}

BENCHMARK(collapsed_icm, 100x200, 30, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace collapsed_icm_bench_vars;

    x = 100;
    y = 200;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params<double>(x, y, r);
    celero::DoNotOptimizeAway(bld::collapsed_icm(X, r, params));
}

BENCHMARK(collapsed_icm, 100x300, 30, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace collapsed_icm_bench_vars;

    x = 100;
    y = 300;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params<double>(x, y, r);
    celero::DoNotOptimizeAway(bld::collapsed_icm(X, r, params));
}

BENCHMARK(collapsed_icm, 100x400, 30, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace collapsed_icm_bench_vars;

    x = 100;
    y = 400;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params<double>(x, y, r);
    celero::DoNotOptimizeAway(bld::collapsed_icm(X, r, params));
}
