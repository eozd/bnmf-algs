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
/**
 * @brief Namespace containing online_EM algorithm related functions.
 */
namespace online_EM {

/**
 * @brief Initialize all NaN values in the given matrix with the mean of the
 * remaining values that are different than NaN.
 *
 * @tparam T Type of the matrix entries.
 * @param X Input matrix containing NaN values.
 * @return Modified matrix containing the mean of non-NaN values in place of
 * every previously NaN value.
 */
template <typename T> size_t init_nan_values(matrix_t<T>& X) {
    const auto x = static_cast<size_t>(X.rows());
    const auto y = static_cast<size_t>(X.cols());

    T nonnan_sum = T();
    size_t nonnan_count = 0;

    #pragma omp parallel for reduction(+:nonnan_sum,nonnan_count)
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            if (not std::isnan(X(i, j))) {
                nonnan_sum += X(i, j);
                ++nonnan_count;
            }
        }
    }

    if (nonnan_count != 0) {
        const T nonnan_mean = nonnan_sum / nonnan_count;

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < x; ++i) {
            for (size_t j = 0; j < y; ++j) {
                if (std::isnan(X(i, j))) {
                    X(i, j) = nonnan_mean;
                }
            }
        }
    }

    return nonnan_count;
}

/**
 * @brief Initialize each entry of alpha and beta matrices with the given
 * model parameters.
 *
 * The returned alpha matrix is of shape \f$x \times z\f$ and beta matrix is of
 * shape \f$z \times y\f$.
 *
 * @tparam Scalar Type of the model parameters.
 * @param param_vec Vector containing an alloc_model::Params object for every
 * hidden dimension.
 * @param y Columns of beta matrix.
 *
 * @return alpha and beta matrices initialized with the given parameter values.
 */
template <typename Scalar>
std::pair<matrix_t<Scalar>, matrix_t<Scalar>>
init_alpha_beta(const std::vector<alloc_model::Params<Scalar>>& param_vec,
                size_t y) {
    const auto x = static_cast<size_t>(param_vec[0].alpha.size());
    const auto z = static_cast<size_t>(param_vec.size());

    matrix_t<Scalar> alpha(x, z);
    matrix_t<Scalar> beta(z, y);

    #pragma omp parallel for schedule(static)
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

/**
 * @brief Initialize S_pjk, S_ipk matrices and S_ppk vector from a Dirichlet
 * distribution with all parameters equal to 1.
 *
 * For every nonzero value \f$X_{ij}\f$ in the input matrix X, this function
 * samples from Dirichlet distribution with \f$z\f$ many 1's as the parameter
 * vector and adds the resulting sample to \f$i^{th}\f$ row of S_ipk matrix,
 * \f$j^{th}\f$ row of S_pjk matrix and to S_ppk vector.
 *
 * @tparam T Type of the matrix entries.
 * @param X_full Input matrix X not containing any NaN values (NaNs must be
 * initialized to some value).
 * @param z Rank of the decomposition (hidden layer count).
 * @param ii Row indices of every nonzero value in the original X input matrix.
 * @param jj Column indices of every nonzero value in the original X input
 * matrix.
 *
 * @return std::tuple of S_pjk, S_ipk and S_ppk.
 */
template <typename T>
std::tuple<matrix_t<T>, matrix_t<T>, vector_t<T>>
init_S_xx(const matrix_t<T>& X_full, size_t z, const std::vector<size_t>& ii,
          const std::vector<size_t>& jj) {
    const auto x = static_cast<size_t>(X_full.rows());
    const auto y = static_cast<size_t>(X_full.cols());

    // results
    matrix_t<T> S_pjk = matrix_t<T>::Zero(y, z);
    matrix_t<T> S_ipk = matrix_t<T>::Zero(x, z);
    vector_t<T> S_ppk = vector_t<T>::Zero(z);

    util::gsl_rng_wrapper rnd_gen(gsl_rng_alloc(gsl_rng_taus), gsl_rng_free);

    vector_t<double> dirichlet_params = vector_t<double>::Constant(z, 1);

    #pragma omp parallel
    {
        // thread local variables
        vector_t<T> fiber(z);
        vector_t<double> fiber_double(z);
        matrix_t<T> S_pjk_local = matrix_t<T>::Zero(y, z);
        matrix_t<T> S_ipk_local = matrix_t<T>::Zero(x, z);
        vector_t<T> S_ppk_local = vector_t<T>::Zero(z);

        // do in parallel
        #pragma omp for
        for (size_t t = 0; t < ii.size(); ++t) {
            size_t i = ii[t], j = jj[t];

            gsl_ran_dirichlet(rnd_gen.get(), z, dirichlet_params.data(),
                              fiber_double.data());
            fiber = fiber_double.template cast<T>();
            fiber = fiber.array() * X_full(i, j);

            S_pjk_local.row(j) += fiber;
            S_ipk_local.row(i) += fiber;
            S_ppk_local += fiber;
        }

        // reduce local variables into accumulators one thread at a time
        #pragma omp critical
        {
            S_pjk += S_pjk_local;
            S_ipk += S_ipk_local;
            S_ppk += S_ppk_local;
        }
    }

    return std::make_tuple(S_pjk, S_ipk, S_ppk);
}

/**
 * @brief Perform an update on logW matrix.
 *
 * @tparam T Type of the input matrix.
 * @tparam Scalar Type of the model parameters.
 * @param alpha alpha matrix returned by init_alpha_beta.
 * @param S_ipk Sum of \f$S\f$ tensor along its second dim, i.e. \f$S_{i+k}\f$.
 * @param alpha_pk Vector containing column sums of alpha matrix.
 * @param S_ppk Sum of \f$S\f$ tensor along its first 2 dimensions, i.e.
 * \f$S_{++k}\f$.
 * @param psi_fn Psi function to use.
 * @param logW logW matrix to update in-place.
 */
template <typename T, typename Scalar>
void update_logW(const matrix_t<Scalar>& alpha, const matrix_t<T>& S_ipk,
                 const vector_t<Scalar>& alpha_pk, const vector_t<T>& S_ppk,
                 const std::function<T(T)>& psi_fn, matrix_t<T>& logW) {
    const auto x = static_cast<size_t>(alpha.rows());
    const auto z = static_cast<size_t>(alpha.cols());

    vector_t<T> psi_of_sums(z);
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < z; ++k) {
        psi_of_sums(k) = psi_fn(alpha_pk(k) + S_ppk(k));
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x; ++i) {
        for (size_t k = 0; k < z; ++k) {
            logW(i, k) = psi_fn(alpha(i, k) + S_ipk(i, k)) - psi_of_sums(k);
        }
    }
}

/**
 * @brief Perform an update on logW matrix.
 *
 * @tparam T Type of the input matrix.
 * @tparam Scalar Type of the model parameters.
 * @param beta beta matrix returned by init_alpha_beta.
 * @param S_pjk Sum of \f$S\f$ along its first dim, i.e. \f$S_{+jk}\f$.
 * @param b b model parameter passed to EM algorithm. (rate parameter of Gamma
 * distribution).
 * @param psi_fn Psi function to use.
 * @param logH logH matrix to update in-place.
 */
template <typename T, typename Scalar>
void update_logH(const matrix_t<Scalar>& beta, const matrix_t<T>& S_pjk,
                 const Scalar b, const std::function<T(T)>& psi_fn,
                 matrix_t<T>& logH) {
    const auto z = static_cast<size_t>(beta.rows());
    const auto y = static_cast<size_t>(beta.cols());

    const Scalar logb = std::log(b + 1);

    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < z; ++k) {
        for (size_t j = 0; j < y; ++j) {
            logH(k, j) = psi_fn(beta(k, j) + S_pjk(j, k)) - logb;
        }
    }
}

/**
 * @brief Find nonzero entries and their indices in the given matrix and return
 * indices and values as vectors.
 *
 * @tparam T Type of the input matrix.
 * @param X Input matrix.
 * @return std::tuple of vectors ii, jj, xx where ii contains row indices, jj
 * contains column indices of the nonzero entries and xx contains the
 * corresponding nonzero entries.
 *
 * @remark If an entry is NaN, it is assumed to be nonzero.
 */
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

/**
 * @brief Update the current allocation by performing a maximization step for
 * each nonzero entry of the original input matrix.
 *
 * @tparam T Type of the input matrix.
 * @param ii Row indices of the original nonzero entries.
 * @param jj Column indices of the original nonzero entries.
 * @param xx Corresponding original nonzero entries.
 * @param res EMResult object containing the results computed up until now.
 * Matrices and sums given in this EMResult object will be modified with the
 * computed values.
 * @param S_ppk S_ppk vector that will be modified in-place.
 *
 * @return Expectation-maximization bound computed in this update.
 */
template <typename T>
T update_allocation(const std::vector<size_t>& ii,
                    const std::vector<size_t>& jj, const std::vector<T>& xx,
                    bld::EMResult<T>& res, vector_t<T>& S_ppk) {

    const auto x = static_cast<size_t>(res.X_full.rows());
    const auto y = static_cast<size_t>(res.X_full.cols());
    const auto z = static_cast<size_t>(S_ppk.cols());

    T EM_bound = T();

    #pragma omp parallel
    {
        matrix_t<T> S_pjk = matrix_t<T>::Zero(y, z);
        matrix_t<T> S_ipk = matrix_t<T>::Zero(x, z);
        vector_t<T> S_ppk_local = vector_t<T>::Zero(z);

        vector_t<T> gammaln_max_fiber(z);
        vector_t<T> log_p(z);
        vector_t<T> max_fiber(z);

        T EM_bound_local = T();

        #pragma omp for
        for (size_t t = 0; t < xx.size(); ++t) {
            const size_t i = ii[t];
            const size_t j = jj[t];
            const T orig_x_ij = xx[t];

            log_p = res.logW.row(i) + res.logH.col(j).transpose();
            const vector_t<T> log_p_exp = log_p.array().exp();

            // maximization step
            if (std::isnan(orig_x_ij)) {
                max_fiber = log_p_exp.array().floor();
                res.X_full(i, j) = max_fiber.sum();
            } else {
                util::multinomial_mode(orig_x_ij, log_p_exp, max_fiber);
            }

            S_pjk.row(j) += max_fiber;
            S_ipk.row(i) += max_fiber;
            S_ppk_local += max_fiber;

            for (size_t k = 0; k < z; ++k) {
                gammaln_max_fiber(k) = gsl_sf_lngamma(max_fiber(k) + 1);
            }
            EM_bound_local +=
                (max_fiber.array() * log_p.array() - gammaln_max_fiber.array())
                    .sum();
        }

        #pragma omp critical
        {
            res.S_pjk += S_pjk;
            res.S_ipk += S_ipk;
            S_ppk += S_ppk_local;
            EM_bound += EM_bound_local;
        }
    }

    return EM_bound;
}

} // namespace online_EM
} // namespace details
} // namespace bnmf_algs