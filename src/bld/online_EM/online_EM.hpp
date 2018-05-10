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

    // init alpha and beta
    matrix_t<Scalar> alpha, beta;
    std::tie(alpha, beta) = details::online_EM::init_alpha_beta(param_vec, y);
    vector_t<Scalar> alpha_pk = alpha.colwise().sum();

    // init S sums
    std::tie(res.S_pjk, res.S_ipk) =
        details::online_EM::init_S_xx(res.X_full, alpha, beta);
    vector_t<T> S_ppk = res.S_ipk.colwise().sum();

    // psi func to use
    std::function<T(T)> psi_fn =
        use_psi_appr ? util::psi_appr<T> : details::gsl_psi_wrapper<T>;

    // init logW and logH
    res.logW = matrix_t<T>(x, z);
    res.logH = matrix_t<T>(z, y);
    details::online_EM::update_logW(alpha, res.S_ipk, alpha_pk, S_ppk, psi_fn,
                                    res.logW);
    details::online_EM::update_logH(beta, res.S_pjk, param_vec[0].b, psi_fn,
                                    res.logH);
    // nonzero elems and indices (including NaN)
    std::vector<size_t> ii, jj;
    std::vector<T> xx;
    std::tie(ii, jj, xx) = details::online_EM::find_nonzero(X);

    // iteration variables
    vector_t<T> gammaln_max_fiber(z);
    vector_t<T> log_p(z);
    vector_t<T> max_fiber(z);
    res.EM_bound = vector_t<T>::Constant(max_iter, 0);

    // EM
    for (size_t iter = 0; iter < max_iter; ++iter) {
        res.S_pjk.setZero();
        res.S_ipk.setZero();
        S_ppk.setZero();

        details::online_EM::update_allocation(
            ii, jj, xx, iter, res, S_ppk, log_p, max_fiber, gammaln_max_fiber);

        details::online_EM::update_logW(alpha, res.S_ipk, alpha_pk, S_ppk,
                                        psi_fn, res.logW);
        details::online_EM::update_logH(beta, res.S_pjk, param_vec[0].b, psi_fn,
                                        res.logH);
    }

    return res;
}
} // namespace bld
} // namespace bnmf_algs
