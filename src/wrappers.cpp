#include "wrappers.hpp"
#include <gsl/gsl_randist.h>

bnmf_algs::gsl_rng_wrapper bnmf_algs::make_gsl_rng(const gsl_rng_type* T) {
    return gsl_rng_wrapper(gsl_rng_alloc(T), gsl_rng_free);
}

bnmf_algs::gsl_ran_discrete_wrapper
bnmf_algs::make_gsl_ran_discrete(size_t K, const double* P) {
    return gsl_ran_discrete_wrapper(gsl_ran_discrete_preproc(K, P),
                                    gsl_ran_discrete_free);
}
