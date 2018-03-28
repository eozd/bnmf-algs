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
}

BASELINE(nmf_seq_elems, KL_0_to_5, 0, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace nmf_seq_elems_bench_vars;

    beg = 0, scale = 5;

    matrix_t X = make_matrix(x, y, beg, scale);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}

BENCHMARK(nmf_seq_elems, KL_0_to_10, 0, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace nmf_seq_elems_bench_vars;

    beg = 0, scale = 10;

    matrix_t X = make_matrix(x, y, beg, scale);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}

BENCHMARK(nmf_seq_elems, KL_0_to_50, 0, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace nmf_seq_elems_bench_vars;

    beg = 0, scale = 50;

    matrix_t X = make_matrix(x, y, beg, scale);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}

BENCHMARK(nmf_seq_elems, KL_0_to_200, 0, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace nmf_seq_elems_bench_vars;

    beg = 0, scale = 200;

    matrix_t X = make_matrix(x, y, beg, scale);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}



BASELINE(nmf_seq_elems, SEQ_0_to_5, 0, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace nmf_seq_elems_bench_vars;

    beg = 0, scale = 5;

    auto params = make_params(x, y, r);
    matrix_t X = make_matrix(x, y, beg, scale);
    celero::DoNotOptimizeAway(bld::seq_greedy_bld(X, r, params));
}

BENCHMARK(nmf_seq_elems, SEQ_0_to_10, 0, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace nmf_seq_elems_bench_vars;

    beg = 0, scale = 10;

    auto params = make_params(x, y, r);
    matrix_t X = make_matrix(x, y, beg, scale);
    celero::DoNotOptimizeAway(bld::seq_greedy_bld(X, r, params));
}

BENCHMARK(nmf_seq_elems, SEQ_0_to_50, 0, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace nmf_seq_elems_bench_vars;

    beg = 0, scale = 50;

    auto params = make_params(x, y, r);
    matrix_t X = make_matrix(x, y, beg, scale);
    celero::DoNotOptimizeAway(bld::seq_greedy_bld(X, r, params));
}

BENCHMARK(nmf_seq_elems, SEQ_0_to_200, 0, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace nmf_seq_elems_bench_vars;

    beg = 0, scale = 20;

    auto params = make_params(x, y, r);
    matrix_t X = make_matrix(x, y, beg, scale);
    celero::DoNotOptimizeAway(bld::seq_greedy_bld(X, r, params));
}