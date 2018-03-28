#include <celero/Celero.h>

#include "bnmf_algs.hpp"
#include <fstream>
#include <iostream>

using namespace bnmf_algs;

CELERO_MAIN

size_t r = 17;
double beta = 1;
size_t max_iter = 1000;

static matrix_t make_matrix(long x, long y, double beg, double scale) {
    matrix_t res = matrix_t::Random(x, y) + matrix_t::Constant(x, y, beg +1);
    res = res.array() * (scale / 2);

    return res;
}

static allocation_model::AllocModelParams make_params(long x, long y) {
    allocation_model::AllocModelParams params(
            100, 1, std::vector<double>(x, 0.05), std::vector<double>(r, 10));
    params.beta.back() = 60;

    return params;
}

matrix_t X;

BASELINE(collapsed_icm, 100x100, 0, 1) {
    X = make_matrix(100, 100, 0, 5);
    auto params = make_params(100, 100);
    celero::DoNotOptimizeAway(bld::collapsed_icm(X, r, params));
}

BENCHMARK(collapsed_icm, 200x100, 0, 1) {
    X = make_matrix(200, 100, 0, 5);
    auto params = make_params(200, 100);
    celero::DoNotOptimizeAway(bld::collapsed_icm(X, r, params));
}

BENCHMARK(collapsed_icm, 300x100, 0, 1) {
    X = make_matrix(300, 100, 0, 5);
    auto params = make_params(300, 100);
    celero::DoNotOptimizeAway(bld::collapsed_icm(X, r, params));
}

BENCHMARK(collapsed_icm, 400x100, 0, 1) {
    X = make_matrix(400, 100, 0, 5);
    auto params = make_params(400, 100);
    celero::DoNotOptimizeAway(bld::collapsed_icm(X, r, params));
}

BENCHMARK(collapsed_icm, 100x200, 0, 1) {
    X = make_matrix(100, 200, 0, 5);
    auto params = make_params(100, 200);
    celero::DoNotOptimizeAway(bld::collapsed_icm(X, r, params));
}

BENCHMARK(collapsed_icm, 100x300, 0, 1) {
    X = make_matrix(100, 300, 0, 5);
    auto params = make_params(100, 300);
    celero::DoNotOptimizeAway(bld::collapsed_icm(X, r, params));
}

BENCHMARK(collapsed_icm, 100x400, 0, 1) {
    X = make_matrix(100, 400, 0, 5);
    auto params = make_params(100, 400);
    celero::DoNotOptimizeAway(bld::collapsed_icm(X, r, params));
}

