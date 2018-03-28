#pragma once

/**
 * This is the benchmark file to test bnmf_algs::bld::bld_mult algorithm
 * performance when matrix size is increased.
 */

#include <celero/Celero.h>

#include "bench_utils.hpp"
#include "bnmf_algs.hpp"
#include <fstream>
#include <iostream>

namespace bld_mult_bench_vars {
long x = 10;
long y = 100;
long beg = 0;
long scale = 5;
size_t r = 17;
} // namespace bld_mult_bench_vars

BASELINE(bld_mult, 100x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_mult_bench_vars;

    x = 100;
    y = 100;

    matrix_t X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_mult(X, r, params));
}

BENCHMARK(bld_mult, 200x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_mult_bench_vars;

    x = 200;
    y = 100;

    matrix_t X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_mult(X, r, params));
}

BENCHMARK(bld_mult, 300x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_mult_bench_vars;

    x = 300;
    y = 100;

    matrix_t X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_mult(X, r, params));
}

BENCHMARK(bld_mult, 400x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_mult_bench_vars;

    x = 400;
    y = 100;

    matrix_t X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_mult(X, r, params));
}

BENCHMARK(bld_mult, 100x200, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_mult_bench_vars;

    x = 100;
    y = 200;

    matrix_t X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_mult(X, r, params));
}

BENCHMARK(bld_mult, 100x300, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_mult_bench_vars;

    x = 100;
    y = 300;

    matrix_t X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_mult(X, r, params));
}

BENCHMARK(bld_mult, 100x400, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_mult_bench_vars;

    x = 100;
    y = 400;

    matrix_t X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_mult(X, r, params));
}
