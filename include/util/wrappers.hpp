#pragma once

#include <gsl/gsl_randist.h>
#include <memory>

namespace bnmf_algs {
namespace util {
/**
 * @brief RAII wrapper around gsl_rng types.
 */
using gsl_rng_wrapper = std::unique_ptr<gsl_rng, decltype(&gsl_rng_free)>;

/**
 * @brief RAII wrapper around gsl_ran_discrete types.
 */
using gsl_ran_discrete_wrapper =
    std::unique_ptr<gsl_ran_discrete_t, decltype(&gsl_ran_discrete_free)>;

/**
 * @brief Construct a new bnmf_algs::util::gsl_rng_wrapper from the given rng
 * type.
 *
 * @param T Random number generator type.
 *
 * @return bnmf_algs::util::gsl_rng_wrapper type containing a gsl_rng in it.
 *
 * @see gsl_rng_alloc.
 */
gsl_rng_wrapper make_gsl_rng(const gsl_rng_type* T);

/**
 * @brief Construct a new bnmf_algs::util::gsl_ran_discrete_wrapper.
 *
 * From gsl_ran_discrete_preproc documentation:
 *
 * <BLOCKQUOTE>
 * The array P contains the probabilities of the discrete events; these array
 * elements must all be positive, but they needn’t add up to one (so you can
 * think of them more generally as “weights”)—the preprocessor will normalize
 * appropriately.
 * </BLOCKQUOTE>
 *
 * @param K Number of objects to choose from.
 * @param P Pointer to an array containing the probability of \f$i^{th}\f$ item
 * at P[i].
 *
 * @return bnmf_algs::util::gsl_ran_discrete_wrapper wrapping a
 * gsl_ran_discrete_t object.
 *
 * @see gsl_ran_discrete_preproc
 */
gsl_ran_discrete_wrapper make_gsl_ran_discrete(size_t K, const double* P);
} // namespace util
} // namespace bnmf_algs
