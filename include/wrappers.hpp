#pragma once

#include <gsl/gsl_randist.h>
#include <memory>

namespace bnmf_algs {
/**
 * @brief RAII wrapper around gsl_rng types.
 */
using gsl_rng_wrapper = std::unique_ptr<gsl_rng, decltype(&gsl_rng_free)>;

/**
 * @brief Construct a new bnmf_algs::gsl_rng_wrapper from the given rng type.
 *
 * @param T Random number generator type.
 *
 * @return bnmf_algs::gsl_rng_wrapper type containing a gsl_rng in it.
 */
gsl_rng_wrapper make_gsl_rng(const gsl_rng_type* T);
} // namespace bnmf_algs
