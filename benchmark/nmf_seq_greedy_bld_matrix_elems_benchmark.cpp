#include <celero/Celero.h>

#include "bnmf_algs.hpp"
#include <fstream>
#include <iostream>

using namespace bnmf_algs;

CELERO_MAIN

size_t x = 100, y = 100;
size_t r = 17;
double beta = 1;
size_t max_iter = 1000;

static matrix_t make_matrix(long x, long y, double beg, double scale) {
    matrix_t res = matrix_t::Random(x, y) + matrix_t::Constant(x, y, beg +1);
    res = res.array() * (scale / 2);

    return res;
}

static allocation_model::AllocModelParams make_params() {
    allocation_model::AllocModelParams params(
        100, 1, std::vector<double>(x, 0.05), std::vector<double>(r, 10));
    params.beta.back() = 60;

    return params;
}

matrix_t X;
auto params = make_params();

BASELINE(kl_nmf, 0_5, 0, 1) {
    X = make_matrix(x, y, 0, 5);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}

BASELINE(seq_greedy_bld, 0_5, 0, 1) {
    X = make_matrix(x, y, 0, 5);
    celero::DoNotOptimizeAway(bld::seq_greedy_bld(X, r, params));
}

BENCHMARK(kl_nmf, 0_10, 0, 1) {
    X = make_matrix(x, y, 0, 10);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}

BENCHMARK(seq_greedy_bld, 0_10, 0, 1) {
    X = make_matrix(x, y, 0, 10);
    celero::DoNotOptimizeAway(bld::seq_greedy_bld(X, r, params));
}

BENCHMARK(kl_nmf, 0_50, 0, 1) {
    X = make_matrix(x, y, 0, 50);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}

BENCHMARK(seq_greedy_bld, 0_50, 0, 1) {
    X = make_matrix(x, y, 0, 50);
    celero::DoNotOptimizeAway(bld::seq_greedy_bld(X, r, params));
}

BENCHMARK(kl_nmf, 0_200, 0, 1) {
    X = make_matrix(x, y, 0, 200);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}

BENCHMARK(seq_greedy_bld, 0_200, 0, 1) {
    X = make_matrix(x, y, 0, 200);
    celero::DoNotOptimizeAway(bld::seq_greedy_bld(X, r, params));
}
