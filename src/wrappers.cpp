#include "wrappers.hpp"
#include <gsl/gsl_randist.h>

bnmf_algs::gsl_rng_wrapper bnmf_algs::make_gsl_rng(const gsl_rng_type* T) {
    return gsl_rng_wrapper(gsl_rng_alloc(T), gsl_rng_free);
}

