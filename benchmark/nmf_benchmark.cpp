#include <celero/Celero.h>

#include "bnmf_algs.hpp"
#include <fstream>
#include <iostream>

using namespace bnmf_algs;

CELERO_MAIN

static matrix_t make_matrix(long x, long y) {
    return matrix_t::Random(x, y) + matrix_t::Constant(x, y, 10);
}

size_t r;
double beta;
size_t max_iter;


BASELINE(100x100, beta_0, 0, 1) {
    r = 15, beta = 0, max_iter = 1000;
    matrix_t X = make_matrix(100, 100);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}

BENCHMARK(100x100, beta_1, 0, 1) {
    r = 15, beta = 1, max_iter = 1000;
    matrix_t X = make_matrix(100, 100);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}

BENCHMARK(100x100, beta_2, 0, 1) {
    r = 15, beta = 2, max_iter = 1000;
    matrix_t X = make_matrix(100, 100);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}



BASELINE(200x100, beta_0, 0, 1) {
    r = 15, beta = 0, max_iter = 1000;
    matrix_t X = make_matrix(200, 100);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}

BENCHMARK(200x100, beta_1, 0, 1) {
    r = 15, beta = 1, max_iter = 1000;
    matrix_t X = make_matrix(200, 100);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}

BENCHMARK(200x100, beta_2, 0, 1) {
    r = 15, beta = 2, max_iter = 1000;
    matrix_t X = make_matrix(200, 100);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}



BASELINE(300x100, beta_0, 0, 1) {
    r = 15, beta = 0, max_iter = 1000;
    matrix_t X = make_matrix(300, 100);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}

BENCHMARK(300x100, beta_1, 0, 1) {
    r = 15, beta = 1, max_iter = 1000;
    matrix_t X = make_matrix(300, 100);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}

BENCHMARK(300x100, beta_2, 0, 1) {
    r = 15, beta = 2, max_iter = 1000;
    matrix_t X = make_matrix(300, 100);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}



BASELINE(400x100, beta_0, 0, 1) {
    r = 15, beta = 0, max_iter = 1000;
    matrix_t X = make_matrix(400, 100);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}

BENCHMARK(400x100, beta_1, 0, 1) {
    r = 15, beta = 1, max_iter = 1000;
    matrix_t X = make_matrix(400, 100);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}

BASELINE(400x100, beta_2, 0, 1) {
    r = 15, beta = 2, max_iter = 1000;
    matrix_t X = make_matrix(400, 100);
    celero::DoNotOptimizeAway(nmf::nmf(X, r, beta, max_iter));
}
