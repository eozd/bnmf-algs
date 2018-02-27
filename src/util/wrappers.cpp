#include "util/wrappers.hpp"
#include <gsl/gsl_randist.h>

using namespace bnmf_algs;

util::gsl_rng_wrapper util::make_gsl_rng(const gsl_rng_type* T) {
    return util::gsl_rng_wrapper(gsl_rng_alloc(T), gsl_rng_free);
}

util::gsl_ran_discrete_wrapper util::make_gsl_ran_discrete(size_t K,
                                                           const double* P) {
    return util::gsl_ran_discrete_wrapper(gsl_ran_discrete_preproc(K, P),
                                          gsl_ran_discrete_free);
}
