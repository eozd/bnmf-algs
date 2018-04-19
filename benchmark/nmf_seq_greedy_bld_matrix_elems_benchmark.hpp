#pragma once

/**
 * This is the benchmark file to compare bnmf_algs::nmf::nmf and
 * bnmf_algs::bld::seq_greedy_bld performance as individual matrix entries get
 * larger.
 */

#include <celero/Celero.h>

#include "bench_utils.hpp"
#include "bnmf_algs.hpp"
#include <fstream>
#include <iostream>

namespace nmf_seq_elems_bench_vars {
long x = 100;
long y = 100;
long r = 17;
double beta = 1;
double beg = 0;
double scale = 5;
size_t max_iter = 100;
} // namespace nmf_seq_elems_bench_vars

BASELINE(nmf_seq_elems, KL_0_to_5, 0, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace nmf_seq_elems_bench_vars;

    beg = 0, scale = 5;

    matrixd X = make_matrix(x, y, beg, scale);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}

BENCHMARK(nmf_seq_elems, KL_0_to_10, 0, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace nmf_seq_elems_bench_vars;

    beg = 0, scale = 10;

    matrixd X = make_matrix(x, y, beg, scale);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}

BENCHMARK(nmf_seq_elems, KL_0_to_50, 0, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace nmf_seq_elems_bench_vars;

    beg = 0, scale = 50;

    matrixd X = make_matrix(x, y, beg, scale);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}

BENCHMARK(nmf_seq_elems, KL_0_to_200, 0, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace nmf_seq_elems_bench_vars;

    beg = 0, scale = 200;

    matrixd X = make_matrix(x, y, beg, scale);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}

BASELINE(nmf_seq_elems, SEQ_0_to_5, 0, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace nmf_seq_elems_bench_vars;

    beg = 0, scale = 5;

    auto params = make_params<double>(x, y, r);
    matrixd X = make_matrix(x, y, beg, scale);
    celero::DoNotOptimizeAway(bld::seq_greedy_bld(X, r, params));
}

BENCHMARK(nmf_seq_elems, SEQ_0_to_10, 0, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace nmf_seq_elems_bench_vars;

    beg = 0, scale = 10;

    auto params = make_params<double>(x, y, r);
    matrixd X = make_matrix(x, y, beg, scale);
    celero::DoNotOptimizeAway(bld::seq_greedy_bld(X, r, params));
}

BENCHMARK(nmf_seq_elems, SEQ_0_to_50, 0, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace nmf_seq_elems_bench_vars;

    beg = 0, scale = 50;

    auto params = make_params<double>(x, y, r);
    matrixd X = make_matrix(x, y, beg, scale);
    celero::DoNotOptimizeAway(bld::seq_greedy_bld(X, r, params));
}

BENCHMARK(nmf_seq_elems, SEQ_0_to_200, 0, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace nmf_seq_elems_bench_vars;

    beg = 0, scale = 20;

    auto params = make_params<double>(x, y, r);
    matrixd X = make_matrix(x, y, beg, scale);
    celero::DoNotOptimizeAway(bld::seq_greedy_bld(X, r, params));
}
