#pragma once

/**
 * This is the benchmark file to test bnmf_algs::bld::collapsed_gibbs algorithm
 * performance when matrix size is increased.
 */

#include <celero/Celero.h>

#include "bench_utils.hpp"
#include "bnmf_algs.hpp"
#include <fstream>
#include <iostream>

namespace collapsed_gibbs_bench_vars {
long x = 100;
long y = 100;
double beg = 0;
double scale = 5;
size_t r = 17;

auto do_nothing = [](const auto& param) -> void {};
} // namespace collapsed_gibbs_bench_vars

BASELINE(collapsed_gibbs, 100x100, 30, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace collapsed_gibbs_bench_vars;

    x = 100;
    y = 100;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params<double>(x, y, r);
    auto gen = bld::collapsed_gibbs(X, r, params);
    celero::DoNotOptimizeAway(
        std::for_each(gen.begin(), gen.end(), do_nothing));
}

BENCHMARK(collapsed_gibbs, 200x100, 30, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace collapsed_gibbs_bench_vars;

    x = 200;
    y = 100;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params<double>(x, y, r);
    auto gen = bld::collapsed_gibbs(X, r, params);
    celero::DoNotOptimizeAway(
        std::for_each(gen.begin(), gen.end(), do_nothing));
}

BENCHMARK(collapsed_gibbs, 300x100, 30, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace collapsed_gibbs_bench_vars;

    x = 300;
    y = 100;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params<double>(x, y, r);
    auto gen = bld::collapsed_gibbs(X, r, params);
    celero::DoNotOptimizeAway(
        std::for_each(gen.begin(), gen.end(), do_nothing));
}

BENCHMARK(collapsed_gibbs, 400x100, 30, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace collapsed_gibbs_bench_vars;

    x = 400;
    y = 100;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params<double>(x, y, r);
    auto gen = bld::collapsed_gibbs(X, r, params);
    celero::DoNotOptimizeAway(
        std::for_each(gen.begin(), gen.end(), do_nothing));
}

BENCHMARK(collapsed_gibbs, 100x200, 30, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace collapsed_gibbs_bench_vars;

    x = 100;
    y = 200;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params<double>(x, y, r);
    auto gen = bld::collapsed_gibbs(X, r, params);
    celero::DoNotOptimizeAway(
        std::for_each(gen.begin(), gen.end(), do_nothing));
}

BENCHMARK(collapsed_gibbs, 100x300, 30, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace collapsed_gibbs_bench_vars;

    x = 100;
    y = 300;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params<double>(x, y, r);
    auto gen = bld::collapsed_gibbs(X, r, params);
    celero::DoNotOptimizeAway(
        std::for_each(gen.begin(), gen.end(), do_nothing));
}

BENCHMARK(collapsed_gibbs, 100x400, 30, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace collapsed_gibbs_bench_vars;

    x = 100;
    y = 400;

    matrixd X = make_matrix(x, y, beg, scale);
    auto params = make_params<double>(x, y, r);
    auto gen = bld::collapsed_gibbs(X, r, params);
    celero::DoNotOptimizeAway(
        std::for_each(gen.begin(), gen.end(), do_nothing));
}
