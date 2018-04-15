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

/**************************** Original bld_mult *******************************/

BASELINE(bld_mult, 100x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_mult_bench_vars;

    x = 100;
    y = 100;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_mult(X, r, params, 1000));
}

BENCHMARK(bld_mult, 200x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_mult_bench_vars;

    x = 200;
    y = 100;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_mult(X, r, params, 1000));
}

BENCHMARK(bld_mult, 300x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_mult_bench_vars;

    x = 300;
    y = 100;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_mult(X, r, params, 1000));
}

BENCHMARK(bld_mult, 400x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_mult_bench_vars;

    x = 400;
    y = 100;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_mult(X, r, params, 1000));
}

BENCHMARK(bld_mult, 100x200, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_mult_bench_vars;

    x = 100;
    y = 200;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_mult(X, r, params, 1000));
}

BENCHMARK(bld_mult, 100x300, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_mult_bench_vars;

    x = 100;
    y = 300;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_mult(X, r, params, 1000));
}

BENCHMARK(bld_mult, 100x400, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_mult_bench_vars;

    x = 100;
    y = 400;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_mult(X, r, params, 1000));
}

/*********************** bld_mult with Approximate psi ************************/

BASELINE(bld_mult_psi_appr, 100x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_mult_bench_vars;

    x = 100;
    y = 100;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_mult(X, r, params, 1000, true));
}

BENCHMARK(bld_mult_psi_appr, 200x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_mult_bench_vars;

    x = 200;
    y = 100;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_mult(X, r, params, 1000, true));
}

BENCHMARK(bld_mult_psi_appr, 300x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_mult_bench_vars;

    x = 300;
    y = 100;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_mult(X, r, params, 1000, true));
}

BENCHMARK(bld_mult_psi_appr, 400x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_mult_bench_vars;

    x = 400;
    y = 100;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_mult(X, r, params, 1000, true));
}

BENCHMARK(bld_mult_psi_appr, 100x200, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_mult_bench_vars;

    x = 100;
    y = 200;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_mult(X, r, params, 1000, true));
}

BENCHMARK(bld_mult_psi_appr, 100x300, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_mult_bench_vars;

    x = 100;
    y = 300;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_mult(X, r, params, 1000, true));
}

BENCHMARK(bld_mult_psi_appr, 100x400, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace bld_mult_bench_vars;

    x = 100;
    y = 400;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::bld_mult(X, r, params, 1000, true));
}
