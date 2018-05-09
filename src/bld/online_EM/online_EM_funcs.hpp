#pragma once

#include "alloc_model/alloc_model_params.hpp"
#include "defs.hpp"
#include "online_EM_defs.hpp"
#include "util/sampling.hpp"
#include <cmath>
#include <gsl/gsl_sf_gamma.h>
#include <utility>
#include <vector>

namespace bnmf_algs {
namespace details {
namespace online_EM {

template <typename T> size_t init_nan_values(matrix_t<T> X_full) {
    const auto x = static_cast<size_t>(X_full.rows());
    const auto y = static_cast<size_t>(X_full.cols());

    T nonnan_sum = T();
    size_t nonnan_count = 0;
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            if (not std::isnan(X_full(i, j))) {
                nonnan_sum += X_full(i, j);
                ++nonnan_count;
            }
        }
    }

    if (nonnan_count != 0) {
        T nonnan_mean = nonnan_sum / nonnan_count;
        for (size_t i = 0; i < x; ++i) {
            for (size_t j = 0; j < y; ++j) {
                if (std::isnan(X_full(i, j))) {
                    X_full(i, j) = nonnan_mean;
                }
            }
        }
    }

    return nonnan_count;
}

template <typename Scalar>
std::pair<matrix_t<Scalar>, matrix_t<Scalar>>
init_alpha_beta(const std::vector<alloc_model::Params<Scalar>>& param_vec,
                size_t y) {
    const auto x = static_cast<size_t>(param_vec[0].alpha.size());
    const auto z = static_cast<size_t>(param_vec.size());

    matrix_t<Scalar> alpha(x, z);
    matrix_t<Scalar> beta(z, y);

    for (size_t k = 0; k < z; ++k) {
        for (size_t i = 0; i < x; ++i) {
            alpha(i, k) = param_vec[k].alpha[i];
        }

        for (size_t j = 0; j < y; ++j) {
            beta(k, j) = param_vec[k].beta[j];
        }
    }

    return std::make_pair(alpha, beta);
}

template <typename T, typename Scalar>
std::pair<matrix_t<T>, matrix_t<T>> init_S_xx(const matrix_t<T>& X_full,
                                              const matrix_t<Scalar>& alpha,
                                              const matrix_t<Scalar>& beta) {
    matrix_t<T> W_0 = util::dirichlet_mat(alpha, 0).template cast<T>();
    matrix_t<T> H_0 = util::dirichlet_mat(beta, 0).template cast<T>();

    matrix_t<T> S_pjk =
        ((W_0.transpose() * X_full).array() * H_0.array()).transpose();
    matrix_t<T> S_ipk = (X_full * H_0.transpose()).array() * W_0.array();

    return std::make_pair(S_pjk, S_ipk);
}

template <typename T, typename Scalar>
void update_logW(const matrix_t<Scalar>& alpha, const matrix_t<T>& S_ipk,
                 const vector_t<Scalar>& alpha_pk, const vector_t<T>& S_ppk,
                 const std::function<T(T)>& psi_fn, matrix_t<T>& logW) {
    const auto x = static_cast<size_t>(alpha.rows());
    const auto z = static_cast<size_t>(alpha.cols());

    vector_t<T> psi_of_sums(z);
    for (size_t k = 0; k < z; ++k) {
        psi_of_sums(k) = psi_fn(alpha_pk(k) + S_ppk(k));
    }

    for (size_t i = 0; i < x; ++i) {
        for (size_t k = 0; k < z; ++k) {
            logW(i, k) = psi_fn(alpha(i, k) + S_ipk(i, k)) - psi_of_sums(k);
        }
    }
}

template <typename T, typename Scalar>
void update_logH(const matrix_t<Scalar>& beta, const matrix_t<T>& S_pjk,
                 const Scalar b, const std::function<T(T)>& psi_fn,
                 matrix_t<T>& logH) {
    const auto z = static_cast<size_t>(beta.rows());
    const auto y = static_cast<size_t>(beta.cols());

    const Scalar logb = std::log(b + 1);
    for (size_t k = 0; k < z; ++k) {
        for (size_t j = 0; j < y; ++j) {
            logH(k, j) = psi_fn(beta(k, j) + S_pjk(j, k)) - logb;
        }
    }
}

template <typename T>
std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<T>>
find_nonzero(const matrix_t<T>& X) {
    std::vector<size_t> ii, jj;
    std::vector<T> xx;

    for (long i = 0; i < X.rows(); ++i) {
        for (long j = 0; j < X.cols(); ++j) {
            T x_ij = X(i, j);
            if (std::isnan(x_ij) || x_ij > std::numeric_limits<T>::epsilon()) {
                ii.push_back(static_cast<size_t>(i));
                jj.push_back(static_cast<size_t>(j));
                xx.push_back(x_ij);
            }
        }
    }

    return std::make_tuple(ii, jj, xx);
}

template <typename T>
void update_allocation(const std::vector<size_t>& ii,
                       const std::vector<size_t>& jj, const std::vector<T>& xx,
                       size_t iter, bld::EMResult<T>& res, vector_t<T>& S_ppk,
                       vector_t<T>& log_p, vector_t<T>& max_fiber,
                       vector_t<T>& gammaln_max_fiber) {

    const auto z = static_cast<size_t>(log_p.cols());
    for (size_t t = 0; t < xx.size(); ++t) {
        const size_t i = ii[t];
        const size_t j = jj[t];
        const T orig_x_ij = xx[t];

        log_p = res.logW.row(i).array() + res.logH.col(j).array();
        const vector_t<T> log_p_exp = log_p.array().exp();

        // maximization step
        if (std::isnan(orig_x_ij)) {
            max_fiber = log_p_exp.array().floor();
            res.X_full(i, j) = max_fiber.sum();
        } else {
            util::multinomial_mode(orig_x_ij, log_p_exp, max_fiber);
        }

        res.S_pjk.row(j) += max_fiber;
        res.S_ipk.row(i) += max_fiber;
        S_ppk = S_ppk.array() + max_fiber.array().sum();

        for (size_t k = 0; k < z; ++k) {
            gammaln_max_fiber(k) = gsl_sf_lngamma(max_fiber(k) + 1);
        }
        res.EM_bound(iter) +=
            (max_fiber.array() * log_p.array() - gammaln_max_fiber.array())
                .sum();
    }
}

} // namespace online_EM
} // namespace details
} // namespace bnmf_algs