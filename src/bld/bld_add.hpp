#pragma once

#include "alloc_model/alloc_model_params.hpp"
#include "defs.hpp"
#include "util_details.hpp"
#include "util/util.hpp"
#include <gsl/gsl_sf_psi.h>

namespace bnmf_algs {
namespace bld {
/**
 * @brief Compute tensor \f$S\f$, the solution of BLD problem \cite
 * kurutmazbayesian, from matrix \f$X\f$ using additive gradient ascent updates.
 *
 * According to Allocation Model \cite kurutmazbayesian,
 *
 * \f[
 * L_j \sim \mathcal{G}(a, b) \qquad W_{:k} \sim \mathcal{D}(\alpha) \qquad
 * H_{:j} \sim \mathcal{D}(\beta) \f]
 *
 * Each entry \f$S_{ijk} \sim \mathcal{PO}(W_{ik}H_{kj}L_j)\f$ and overall \f$X
 * = S_{ij+}\f$.
 *
 * In this context, Best Latent Decomposition (BLD) problem is \cite
 * kurutmazbayesian,
 *
 * \f[
 * S^* = \underset{S_{::+}=X}{\arg \max}\text{ }p(S).
 * \f]
 *
 * Additive BLD algorithm solves a constrained optimization problem over
 * the set of tensors whose sum over their third index is equal to \f$X\f$.
 * Let's denote this set with \f$\Omega_X = \{S | S_{ij+} = X_{ij}\}\f$. To make
 * the optimization easier the constraints on the found tensors are relaxed by
 * searching over \f$\mathcal{R}^{x \times y \times z}\f$ and defining a
 * projection function \f$\phi : \mathcal{R}^{x \times y \times z} \rightarrow
 * \Omega_X\f$ such that \f$\phi(Z) = S\f$ where \f$Z\f$ is the solution of the
 * unconstrained problem and \f$S\f$ is the solution of the original constrained
 * problem.
 *
 * In this algorithm, \f$\phi\f$ is chosen such that
 *
 * \f[
 *     S_{ijk} = \phi(Z_{ijk}) =
 * X_{ij}\frac{\exp(Z_{ijk})}{\sum_{k'}\exp(Z_{ijk'})} \f]
 *
 * This particular choice of \f$\phi\f$ leads to additive update rules
 * implemented in this algorithm.
 *
 * @tparam T Type of the matrix/tensor entries.
 * @tparam Scalar Type of the scalars used as model parameters.
 * @param X Nonnegative matrix of size \f$x \times y\f$ to decompose.
 * @param z Number of matrices into which matrix \f$X\f$ will be decomposed.
 * This is the depth of the output tensor \f$S\f$.
 * @param model_params Allocation model parameters. See
 * bnmf_algs::alloc_model::Params<double>.
 * @param max_iter Maximum number of iterations.
 * @param eps Floating point epsilon value to be used to prevent division by 0
 * errors.
 *
 * @return Tensor \f$S\f$ of size \f$x \times y \times z\f$ where \f$X =
 * S_{ij+}\f$.
 *
 * @throws std::invalid_argument if X contains negative entries,
 * if number of rows of X is not equal to number of alpha parameters, if z is
 * not equal to number of beta parameters.
 */
template <typename T, typename Scalar>
tensor_t<T, 3> bld_add(const matrix_t<T>& X, size_t z,
                       const alloc_model::Params<Scalar>& model_params,
                       size_t max_iter = 1000, double eps = 1e-50) {
    details::check_bld_params(X, z, model_params);

    long x = X.rows(), y = X.cols();
    tensor_t<T, 3> S(x, y, z);
    // initialize tensor S
    {
        util::gsl_rng_wrapper rand_gen(gsl_rng_alloc(gsl_rng_taus),
                                       gsl_rng_free);
        std::vector<double> dirichlet_params(z, 1);
        std::vector<double> dirichlet_variates(z);
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                gsl_ran_dirichlet(rand_gen.get(), z, dirichlet_params.data(),
                                  dirichlet_variates.data());
                for (size_t k = 0; k < z; ++k) {
                    S(i, j, k) = X(i, j) * dirichlet_variates[k];
                }
            }
        }
    }

    tensor_t<T, 2> S_pjk(y, z); // S_{+jk}
    tensor_t<T, 2> S_ipk(x, z); // S_{i+k}
    matrix_t<T> alpha_eph(x, z);
    matrix_t<T> beta_eph(y, z);
    tensor_t<T, 3> grad_S(x, y, z);
    tensor_t<T, 3> eps_tensor(x, y, z);
    eps_tensor.setConstant(eps);

    const auto eta = [](const size_t step) -> double {
        return 0.1 / std::pow(step + 1, 0.55);
    };

    for (size_t eph = 0; eph < max_iter; ++eph) {
        // update S_pjk, S_ipk, S_ijp
        S_pjk = S.sum(shape<1>({0}));
        S_ipk = S.sum(shape<1>({1}));
        // update alpha_eph
        for (int i = 0; i < x; ++i) {
            for (size_t k = 0; k < z; ++k) {
                alpha_eph(i, k) = model_params.alpha[i] + S_ipk(i, k);
            }
        }
        // update beta_eph
        for (int j = 0; j < y; ++j) {
            for (size_t k = 0; k < z; ++k) {
                beta_eph(j, k) = model_params.beta[k] + S_pjk(j, k);
            }
        }
        // update grad_S
        vector_t<T> alpha_eph_sum = alpha_eph.colwise().sum();
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                for (size_t k = 0; k < z; ++k) {
                    grad_S(i, j, k) = gsl_sf_psi(beta_eph(j, k)) -
                                      gsl_sf_psi(S(i, j, k) + 1) -
                                      gsl_sf_psi(alpha_eph_sum(k)) +
                                      gsl_sf_psi(alpha_eph(i, k));
                }
            }
        }
        // construct Z
        tensor_t<T, 3> Z = S.log() + eps_tensor;
        tensor_t<T, 2> S_mult = (S * grad_S).sum(shape<1>({2}));
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                for (size_t k = 0; k < z; ++k) {
                    Z(i, j, k) += eta(eph) * S(i, j, k) *
                                  (X(i, j) * grad_S(i, j, k) - S_mult(i, j));
                }
            }
        }
        Z = Z.exp();
        bnmf_algs::util::normalize(Z, 2, bnmf_algs::util::NormType::L1);
        // update S
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                for (size_t k = 0; k < z; ++k) {
                    S(i, j, k) = X(i, j) * Z(i, j, k);
                }
            }
        }
    }
    return S;
}
} // namespace bld
} // namespace bnmf_algs
