#pragma once

#include "alloc_model/alloc_model_params.hpp"
#include "defs.hpp"
#include <gsl/gsl_sf_psi.h>

namespace bnmf_algs {
namespace details {
/**
 * @brief Do parameter checks on BLD computing function parameters and throw an
 * assertion error if the parameters are incorrect.
 *
 * @tparam T Type of the matrix/tensor entries.
 * @tparam Scalar Type of the scalars used as model parameters.
 *
 * @remark Throws assertion error if the parameters are incorrect.
 */
template <typename T, typename Scalar>
void check_bld_params(const matrix_t<T>& X, size_t z,
                      const alloc_model::Params<Scalar>& model_params) {
    BNMF_ASSERT((X.array() >= 0).all(), "X must be nonnegative");
    BNMF_ASSERT(
        model_params.alpha.size() == static_cast<size_t>(X.rows()),
        "Number of alpha parameters must be equal to number of rows of X");
    BNMF_ASSERT(model_params.beta.size() == static_cast<size_t>(z),
                "Number of beta parameters must be equal to z");
}

/**
 * @brief A template wrapper function around gsl_sf_psi.
 *
 * gsl_sf_psi can only be called with double arguments. This function wraps
 * gsl_sf_psi using static casts. Argument type is cast to double, gsl_sf_psi is
 * called and the resulting double is cast back to argument type.
 *
 * @tparam Real Type of the input.
 * @param x Input.
 * @return Output of gsl_sf_psi with the same type as the parameter.
 */
template <typename Real> Real gsl_psi_wrapper(Real x) {
    return static_cast<Real>(gsl_sf_psi(static_cast<double>(x)));
}
} // namespace details
} // namespace bnmf_algs
