#pragma once

/**
 * This is the benchmark file to test util::psi_appr algorithm performance.
 */

#include <celero/Celero.h>

#include "bnmf_algs.hpp"
#include <gsl/gsl_sf_psi.h>

BASELINE(psi_appr, x_00001, 30, 100000) {
    double x = 1e-6;
    celero::DoNotOptimizeAway(bnmf_algs::util::psi_appr(x));
}

BENCHMARK(psi_appr, x_0001, 30, 100000) {
    double x = 1e-4;
    celero::DoNotOptimizeAway(bnmf_algs::util::psi_appr(x));
}

BENCHMARK(psi_appr, x_01, 30, 100000) {
    double x = 1e-2;
    celero::DoNotOptimizeAway(bnmf_algs::util::psi_appr(x));
}

BENCHMARK(psi_appr, x_1, 30, 100000) {
    double x = 1;
    celero::DoNotOptimizeAway(bnmf_algs::util::psi_appr(x));
}

BENCHMARK(psi_appr, x_100, 30, 100000) {
    double x = 100;
    celero::DoNotOptimizeAway(bnmf_algs::util::psi_appr(x));
}
