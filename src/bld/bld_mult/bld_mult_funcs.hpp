#pragma once

#include "alloc_model/alloc_model_params.hpp"
#include "defs.hpp"

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace bnmf_algs {
namespace details {
/**
 * @brief Namespace containing bld_mult update functions (CUDA/non-CUDA).
 */
namespace bld_mult {

/**
 * @brief Initialize S tensor using a Dirichlet sample of size z with all
 * concentration parameters set to 1.
 *
 * S tensor is initialized according to the following update
 *
 * \f[
 *     S_{ijk} \leftarrow X_{ij}\mathcal{D}_{ij}(k)
 * \f]
 *
 * where \f$\mathcal{D}_{ij}(k)\f$ is the kth entry of the \f$ij^{th}\f$
 * Dirichlet sample.
 *
 * @tparam T Type of the entries of the matrix X and tensor S
 * @param X X matrix sent as input to bld algorithm.
 * @param z Rank of the decomposition.
 * @return S_{x \times y \times z} tensor for a matrix \f$X_{x \times y}\f$
 * and rank parameter z.
 */
template <typename T> tensor_t<T, 3> init_S(const matrix_t<T>& X, size_t z) {
    const auto x = static_cast<size_t>(X.rows());
    const auto y = static_cast<size_t>(X.cols());

#ifdef USE_OPENMP
    const int num_threads = std::thread::hardware_concurrency();
#endif

    tensor_t<T, 3> S(x, y, z);
    vectord dirichlet_params = vectord::Constant(z, 1);
    #pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread_id = omp_get_thread_num();
        bool last_thread = thread_id + 1 == num_threads;
        long num_rows = x / num_threads;
        long beg = thread_id * num_rows;
        long end = !last_thread ? beg + num_rows : x;
#else
        long beg = 0;
        long end = x;
#endif

        util::gsl_rng_wrapper rand_gen(gsl_rng_alloc(gsl_rng_taus),
                                       gsl_rng_free);
        vectord dirichlet_variates(z);
        for (long i = beg; i < end; ++i) {
            for (size_t j = 0; j < y; ++j) {
                gsl_ran_dirichlet(rand_gen.get(), z, dirichlet_params.data(),
                                  dirichlet_variates.data());
                for (size_t k = 0; k < z; ++k) {
                    S(i, j, k) = X(i, j) * dirichlet_variates(k);
                }
            }
        }
    }

    return S;
}

/**
 * @brief Compute the reciprocal \f$\hat{X}\f$ of the input matrix \f$X\f$.
 *
 * \f$\hat{X}_{ij} = \frac{1}{X_{ij} + \epsilon}\f$.
 *
 * @tparam T Type of the entries of the input matrix.
 * @param X Input matrix.
 * @param eps Epsilon value to be used to prevent division by 0 errors.
 * @return Reciprocal of the input matrix.
 */
template <typename T>
matrix_t<T> X_reciprocal(const matrix_t<T>& X, double eps) {
    const auto x = static_cast<size_t>(X.rows());
    const auto y = static_cast<size_t>(X.cols());

    matrix_t<T> X_reciprocal(x, y);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            X_reciprocal(i, j) = 1.0 / (X(i, j) + eps);
        }
    }

    return X_reciprocal;
}

/**
 * @brief Initialize alpha and beta vectors used in bld_mult.
 *
 * @tparam Scalar Type of the entries of model parameters.
 * @param params Allocation model parameters.
 * @return std::pair of alpha and beta vectors which are copies of the model
 * parameter alpha and beta vectors.
 */
template <typename Scalar>
std::pair<vector_t<Scalar>, vector_t<Scalar>>
init_alpha_beta(const alloc_model::Params<Scalar>& params) {
    vector_t<Scalar> alpha(params.alpha.size());
    vector_t<Scalar> beta(params.beta.size());
    std::copy(params.alpha.begin(), params.alpha.end(), alpha.data());
    std::copy(params.beta.begin(), params.beta.end(), beta.data());

    return std::make_pair(alpha, beta);
}

/**
 * @brief Update alpha_eph matrix used in bld_mult.
 *
 * @tparam T Type of the matrix entries.
 * @param S_ipk Sum of S tensor along its 1st axis (0 based indexing)
 * @param alpha alpha vector used in bld_mult
 * @param alpha_eph alpha_eph matrix to update.
 */
template <typename T>
void update_alpha_eph(const tensor_t<T, 2>& S_ipk, const vector_t<T>& alpha,
                      matrix_t<T>& alpha_eph) {
    const auto x = static_cast<size_t>(S_ipk.dimension(0));
    const auto z = static_cast<size_t>(S_ipk.dimension(1));

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x; ++i) {
        for (size_t k = 0; k < z; ++k) {
            alpha_eph(i, k) = S_ipk(i, k) + alpha(i);
        }
    }
}

/**
 * @brief Update beta_eph matrix used in bld_mult.
 *
 * @tparam T Type of the matrix entries.
 * @param S_pjk Sum of S tensor along its 0th axis (0 based indexing)
 * @param beta beta vector used in bld_mult
 * @param beta_eph beta_eph matrix to update.
 */
template <typename T>
void update_beta_eph(const tensor_t<T, 2>& S_pjk, const vector_t<T>& beta,
                     matrix_t<T>& beta_eph) {
    const auto y = static_cast<size_t>(S_pjk.dimension(0));
    const auto z = static_cast<size_t>(S_pjk.dimension(1));

    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < y; ++j) {
        for (size_t k = 0; k < z; ++k) {
            beta_eph(j, k) = S_pjk(j, k) + beta(k);
        }
    }
}

/**
 * @brief Update grad_plus tensor used in bld_mult.
 *
 * @tparam T Type of the matrix/tensor entries
 * @tparam PsiFunction A callable type that will return \f$\psi(x)\f$ for an
 * input parameter \f$x\f$.
 * @param S S tensor (output of bld_mult).
 * @param beta_eph beta_eph matrix used in bld_mult.
 * @param psi_fn Psi computing callable.
 * @param grad_plus grad_plus tensor to update.
 */
template <typename T, typename PsiFunction>
void update_grad_plus(const tensor_t<T, 3>& S, const matrix_t<T>& beta_eph,
                      PsiFunction psi_fn, tensor_t<T, 3>& grad_plus) {
    const auto x = static_cast<size_t>(S.dimension(0));
    const auto y = static_cast<size_t>(S.dimension(1));
    const auto z = static_cast<size_t>(S.dimension(2));

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            for (size_t k = 0; k < z; ++k) {
                grad_plus(i, j, k) =
                    psi_fn(beta_eph(j, k)) - psi_fn(S(i, j, k) + 1);
            }
        }
    }
}

/**
 * @brief Update grad_minus tensor used in bld_mult.
 *
 * @tparam T Type of the matrix/tensor entries
 * @tparam PsiFunction A callable type that will return \f$\psi(x)\f$ for an
 * input parameter \f$x\f$.
 * @param alpha_eph alpha_matrix used in bld_mult.
 * @param psi_fn Psi computing callable.
 * @param grad_minus grad_minus tensor to update.
 */
template <typename T, typename PsiFunction>
void update_grad_minus(const matrix_t<T>& alpha_eph, PsiFunction psi_fn,
                       matrix_t<T>& grad_minus) {

    const auto x = static_cast<size_t>(grad_minus.rows());
    const auto z = static_cast<size_t>(grad_minus.cols());

    // update alpha_eph_sum
    vector_t<T> psi_alpha_eph_sum(z);
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < z; ++k) {
        T sum = T();
        for (size_t i = 0; i < x; ++i) {
            sum += alpha_eph(i, k);
        }
        psi_alpha_eph_sum(k) = psi_fn(sum);
    }

    // update grad_minus
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x; ++i) {
        for (size_t k = 0; k < z; ++k) {
            grad_minus(i, k) = psi_alpha_eph_sum(k) - psi_fn(alpha_eph(i, k));
        }
    }
}

/**
 * @brief Update nom_mult matrix used in bld_mult.
 *
 * @tparam T Type of the matrix entries.
 * @param X_reciprocal Matrix containing reciprocals of entries of input matrix
 * X.
 * @param grad_minus grad_minus matrix.
 * @param S S tensor
 * @param nom_mult nom_mult matrix to update.
 */
template <typename T>
void update_nom_mult(const matrix_t<T>& X_reciprocal,
                     const matrix_t<T>& grad_minus, const tensor_t<T, 3>& S,
                     matrix_t<T>& nom_mult) {
    const auto x = static_cast<size_t>(S.dimension(0));
    const auto y = static_cast<size_t>(S.dimension(1));
    const auto z = static_cast<size_t>(S.dimension(2));

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            T nom_sum = T();
            for (size_t k = 0; k < z; ++k) {
                nom_sum += (S(i, j, k) * X_reciprocal(i, j) * grad_minus(i, k));
            }
            nom_mult(i, j) = nom_sum;
        }
    }
}

/**
 * @brief Update denom_mult matrix used in bld_mult.
 *
 * @tparam T Type of the matrix entries.
 * @param X_reciprocal Matrix containing reciprocals of entries of input matrix
 * X.
 * @param grad_plus grad_plus tensor.
 * @param S S tensor
 * @param denom_mult denom_mult matrix to update.
 */
template <typename T>
void update_denom_mult(const matrix_t<T>& X_reciprocal,
                       const tensor_t<T, 3>& grad_plus, const tensor_t<T, 3>& S,
                       matrix_t<T>& denom_mult) {
    const auto x = static_cast<size_t>(S.dimension(0));
    const auto y = static_cast<size_t>(S.dimension(1));
    const auto z = static_cast<size_t>(S.dimension(2));

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            T denom_sum = T();
            for (size_t k = 0; k < z; ++k) {
                denom_sum +=
                    (S(i, j, k) * X_reciprocal(i, j) * grad_plus(i, j, k));
            }
            denom_mult(i, j) = denom_sum;
        }
    }
}

/**
 * @brief Update S tensor (output of bld_mult algorithm).
 *
 * @tparam T Type of tensor entries.
 * @param X Input matrix passed to bld_mult algorithm.
 * @param nom nom_mult matrix used in bld_mult algorithm.
 * @param denom denom_mult matrix used in bld_mult algorithm.
 * @param grad_minus grad_minus matrix used in bld_mult algorithm.
 * @param grad_plus grad_plus tensor used in bld_mult algorithm.
 * @param S_ijp Matrix computed by summing tensor S along its 2nd axis (0 based
 * indexing).
 * @param S S tensor to update.
 * @param eps Epsilon value to prevent division by 0 errors.
 */
template <typename T>
void update_S(const matrix_t<T>& X, const matrix_t<T>& nom,
              const matrix_t<T>& denom, const matrix_t<T>& grad_minus,
              const tensor_t<T, 3>& grad_plus, const tensor_t<T, 2>& S_ijp,
              tensor_t<T, 3>& S, double eps) {
    const auto x = static_cast<size_t>(S.dimension(0));
    const auto y = static_cast<size_t>(S.dimension(1));
    const auto z = static_cast<size_t>(S.dimension(2));

    // update S
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            T s_ij = S_ijp(i, j);
            T x_over_s = X(i, j) / (s_ij + eps);
            for (size_t k = 0; k < z; ++k) {
                S(i, j, k) *= (grad_plus(i, j, k) + nom(i, j)) * x_over_s /
                              (grad_minus(i, k) + denom(i, j) + eps);
            }
        }
    }
}

} // namespace bld_mult
} // namespace details
} // namespace bnmf_algs
