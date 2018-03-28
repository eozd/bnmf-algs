#pragma once

#include <celero/Celero.h>

#include "bench_utils.hpp"
#include "bnmf_algs.hpp"
#include <fstream>
#include <iostream>

namespace collapsed_icm_bench_vars {
    long x = 100;
    long y = 100;
    long beg = 0;
    long scale = 5;
    size_t r = 17;
} // namespace collapsed_icm_bench_vars


BASELINE(collapsed_icm, 100x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace collapsed_icm_bench_vars;

    x = 100;
    y = 100;

    matrix_t X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::collapsed_icm(X, r, params));
}

BENCHMARK(collapsed_icm, 200x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace collapsed_icm_bench_vars;

    x = 200;
    y = 100;

    matrix_t X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::collapsed_icm(X, r, params));
}

BENCHMARK(collapsed_icm, 300x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace collapsed_icm_bench_vars;

    x = 300;
    y = 100;

    matrix_t X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::collapsed_icm(X, r, params));
}

BENCHMARK(collapsed_icm, 400x100, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace collapsed_icm_bench_vars;

    x = 400;
    y = 100;

    matrix_t X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::collapsed_icm(X, r, params));
}

BENCHMARK(collapsed_icm, 100x200, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace collapsed_icm_bench_vars;

    x = 100;
    y = 200;

    matrix_t X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::collapsed_icm(X, r, params));
}

BENCHMARK(collapsed_icm, 100x300, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace collapsed_icm_bench_vars;

    x = 100;
    y = 300;

    matrix_t X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::collapsed_icm(X, r, params));
}

BENCHMARK(collapsed_icm, 100x400, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace collapsed_icm_bench_vars;

    x = 100;
    y = 400;

    matrix_t X = make_matrix(x, y, beg, scale);
    auto params = make_params(x, y, r);
    celero::DoNotOptimizeAway(bld::collapsed_icm(X, r, params));
}

