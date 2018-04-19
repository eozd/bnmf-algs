#pragma once

/**
 * This is the benchmark file to test bnmf_algs::bld::bld_appr algorithm
 * performance when matrix size is increased.
 */

#include <celero/Celero.h>

#include "bench_utils.hpp"
#include "bnmf_algs.hpp"
#include <fstream>
#include <iostream>

namespace bld_appr_bench_vars {
long x = 10;
long y = 100;
double beg = 0;
double scale = 5;
size_t r = 17;
} // namespace bld_appr_bench_vars

BASELINE(bld_appr, 100x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_appr_bench_vars;

    x = 100;
    y = 100;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params<double>(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_appr(X, r, params));
}

BENCHMARK(bld_appr, 200x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_appr_bench_vars;

    x = 200;
    y = 100;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params<double>(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_appr(X, r, params));
}

BENCHMARK(bld_appr, 300x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_appr_bench_vars;

    x = 300;
    y = 100;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params<double>(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_appr(X, r, params));
}

BENCHMARK(bld_appr, 400x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_appr_bench_vars;

    x = 400;
    y = 100;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params<double>(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_appr(X, r, params));
}

BENCHMARK(bld_appr, 100x200, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_appr_bench_vars;

    x = 100;
    y = 200;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params<double>(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_appr(X, r, params));
}

BENCHMARK(bld_appr, 100x300, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_appr_bench_vars;

    x = 100;
    y = 300;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params<double>(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_appr(X, r, params));
}

BENCHMARK(bld_appr, 100x400, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_appr_bench_vars;

    x = 100;
    y = 400;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params<double>(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_appr(X, r, params));
}
