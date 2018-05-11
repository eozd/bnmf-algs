#pragma once

#include "alloc_model/alloc_model_params.hpp"
#include "bld/util_details.hpp"
#include "defs.hpp"
#include "online_EM_defs.hpp"
#include "online_EM_funcs.hpp"
#include "util/util.hpp"
#include <cmath>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_psi.h>
#include <tuple>

namespace bnmf_algs {
namespace bld {

/**
 * @brief Complete a matrix containing unobserved values given as NaN using an
 * EM procedure according to the allocation model proposed by
 * \cite kurutmazbayesian .
 *
 * This procedure completes a matrix \f$X\f$ containing unobserved values as
 * NaNs. The completion is performed using an EM procedure derived according to
 * the allocation model. The hidden tensor described in \cite kurutmazbayesian
 * is never stored explicitly; only its sums along its each dimension is stored.
 * Therefore, space complexity of the method is \f$O(x \times y)\f$ for an
 * input matrix \f$X_{x \times y}\f$.
 *
 * @tparam T Type of the matrix entries.
 * @tparam Scalar Type of the parameters.
 * @param X Incomplete matrix containing NaN values for every unobserved entry.
 * @param param_vec Parameter vector of size \f$z\f$, the rank of the
 * decomposition. Each entry of the parameter vector must be an
 * alloc_model::Params object containing \f$x\f$ many \f$\alpha\f$ and \f$y\f$
 * many \f$\beta\f$ values for the input matrix \f$X_{x \times y}\f$. Note that
 * for this procedure, the rank of the decomposition is not given explicitly; it
 * is inferred as the size of the parameter vector.
 * @param max_iter Maximum number of iterations to run the EM procedure.
 * @param use_psi_appr If true, use util::psi_appr function to compute the
 * digamma function approximately. If false, compute the digamma function
 * exactly.
 *
 * @return Results of the EM procedure as EMResult. See EMResult documentation
 * for explanations of each result.
 *
 * @remark Throws assertion error if matrix X contains negative values, if the
 * given parameter vector is not compatible with the given matrix.
 */
template <typename T, typename Scalar>
EMResult<T> online_EM(const matrix_t<T>& X,
                      const std::vector<alloc_model::Params<Scalar>>& param_vec,
                      const size_t max_iter = 1000,
                      const bool use_psi_appr = false) {
    details::check_EM_params(X, param_vec);

    const auto x = static_cast<size_t>(X.rows());
    const auto y = static_cast<size_t>(X.cols());
    const auto z = static_cast<size_t>(param_vec.size());

    EMResult<T> res;
    res.X_full = X;

    if (details::online_EM::init_nan_values(res.X_full) == 0) {
        // all the values are NaN
        return res;
    }

    // nonzero elems and indices (including NaN)
    // these are const, but no good way of making them so in C++14
    /* const */ std::vector<size_t> ii, jj;
    /* const */ std::vector<T> xx;
    std::tie(ii, jj, xx) = details::online_EM::find_nonzero(X);

    // init alpha and beta
    // these are const, but no good way of making them so in C++14
    /* const */ matrix_t<Scalar> alpha, beta;
    std::tie(alpha, beta) = details::online_EM::init_alpha_beta(param_vec, y);
    const vector_t<Scalar> alpha_pk = alpha.colwise().sum();

    // init S sums
    vector_t<T> S_ppk;
    std::tie(res.S_pjk, res.S_ipk, S_ppk) =
        details::online_EM::init_S_xx(res.X_full, z, ii, jj);

    // psi func to use
    std::function<T(T)> psi_fn =
        use_psi_appr ? util::psi_appr<T> : details::gsl_psi_wrapper<T>;

    // init logW and logH
    res.logW = matrix_t<double>(x, z);
    res.logH = matrix_t<double>(z, y);
    details::online_EM::update_logW(alpha, res.S_ipk, alpha_pk, S_ppk, psi_fn,
                                    res.logW);
    details::online_EM::update_logH(beta, res.S_pjk, param_vec[0].b, psi_fn,
                                    res.logH);

    // iteration variables
    res.log_PS = vector_t<double>::Constant(max_iter, 0);

    // EM
    for (size_t iter = 0; iter < max_iter; ++iter) {
        res.S_pjk.setZero();
        res.S_ipk.setZero();
        S_ppk.setZero();

        double delta_log_PS =
            details::online_EM::update_allocation(ii, jj, xx, res, S_ppk);
        res.log_PS(iter) += delta_log_PS;

        details::online_EM::update_logW(alpha, res.S_ipk, alpha_pk, S_ppk,
                                        psi_fn, res.logW);
        details::online_EM::update_logH(beta, res.S_pjk, param_vec[0].b, psi_fn,
                                        res.logH);

        delta_log_PS = details::online_EM::delta_log_PS(
            alpha, beta, res.S_ipk, res.S_pjk, alpha_pk, S_ppk, param_vec[0].b);
        res.log_PS(iter) += delta_log_PS;
    }

    return res;
}

} // namespace bld
} // namespace bnmf_algs
