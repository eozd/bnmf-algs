#pragma once

#include "alloc_model/alloc_model_params.hpp"
#include "defs.hpp"
#include "util_details.hpp"

namespace bnmf_algs {
namespace details {

template <typename T>
void update_X_hat(const matrix_t<T>& nu, const matrix_t<T>& mu,
                  matrix_t<T>& X_hat) {
    X_hat = nu * mu;
}

template <typename T>
void update_orig_over_appr(const matrix_t<T>& X, const matrix_t<T>& X_hat,
                           const matrix_t<T>& eps_mat,
                           matrix_t<T>& orig_over_appr) {
    orig_over_appr = X.array() / (X_hat + eps_mat).array();
}

template <typename T>
void update_orig_over_appr_squared(const matrix_t<T>& X,
                                   const matrix_t<T>& X_hat,
                                   const matrix_t<T>& eps_mat,
                                   matrix_t<T>& orig_over_appr_squared) {
    matrixd appr_squared = X_hat.array().pow(2);
    orig_over_appr_squared = X.array() / (appr_squared + eps_mat).array();
}

template <typename T>
void update_S(const matrix_t<T>& orig_over_appr, const matrix_t<T>& nu,
              const matrix_t<T>& mu, tensor_t<T, 3>& S) {
    auto x = static_cast<size_t>(S.dimension(0));
    auto y = static_cast<size_t>(S.dimension(1));
    auto z = static_cast<size_t>(S.dimension(2));

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            for (size_t k = 0; k < z; ++k) {
                S(i, j, k) = orig_over_appr(i, j) * nu(i, k) * mu(k, j);
            }
        }
    }
}

template <typename T>
void update_alpha_eph(const vector_t<T>& alpha, const tensor_t<T, 3>& S,
                      matrix_t<T>& alpha_eph) {
    auto x = static_cast<size_t>(S.dimension(0));
    auto z = static_cast<size_t>(S.dimension(2));

    tensor_t<T, 2> S_ipk_tensor(x, z);
    S_ipk_tensor = S.sum(shape<1>({1}));
    Eigen::Map<matrix_t<T>> S_ipk(S_ipk_tensor.data(), x, z);
    alpha_eph = S_ipk.array().colwise() + alpha.transpose().array();
}

template <typename T>
void update_beta_eph(const vector_t<T>& beta, const tensor_t<T, 3>& S,
                     matrix_t<T>& beta_eph) {
    auto y = static_cast<size_t>(S.dimension(1));
    auto z = static_cast<size_t>(S.dimension(2));

    tensor_t<T, 2> S_pjk_tensor = S.sum(shape<1>({0}));
    Eigen::Map<matrix_t<T>> S_pjk(S_pjk_tensor.data(), y, z);
    beta_eph = S_pjk.array().rowwise() + beta.array();
}

template <typename T>
void update_grad_plus(const matrix_t<T>& beta_eph, const tensor_t<T, 3>& S,
                      tensor_t<T, 3>& grad_plus) {
    auto x = static_cast<size_t>(grad_plus.dimension(0));
    auto y = static_cast<size_t>(grad_plus.dimension(1));
    auto z = static_cast<size_t>(grad_plus.dimension(2));

    #pragma omp parallel for
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            for (size_t k = 0; k < z; ++k) {
                grad_plus(i, j, k) =
                    util::psi_appr(beta_eph(j, k)) - util::psi_appr(S(i, j, k) + 1);
            }
        }
    }
}

template <typename T>
void update_grad_minus(const matrix_t<T>& alpha_eph, matrix_t<T>& grad_minus) {
    auto x = static_cast<size_t>(grad_minus.rows());
    auto z = static_cast<size_t>(grad_minus.cols());

    vector_t<T> alpha_eph_sum = alpha_eph.colwise().sum();
    #pragma omp parallel for
    for (size_t i = 0; i < x; ++i) {
        for (size_t k = 0; k < z; ++k) {
            grad_minus(i, k) =
                util::psi_appr(alpha_eph_sum(k)) - util::psi_appr(alpha_eph(i, k));
        }
    }
}

template <typename T>
void update_nu(const matrix_t<T>& orig_over_appr,
               const matrix_t<T>& orig_over_appr_squared, const matrix_t<T>& mu,
               const tensor_t<T, 3>& grad_plus, const matrix_t<T>& grad_minus,
               double eps, matrix_t<T>& nu) {
    auto x = static_cast<size_t>(grad_plus.dimension(0));
    auto y = static_cast<size_t>(grad_plus.dimension(1));
    auto z = static_cast<size_t>(grad_plus.dimension(2));

    matrix_t<T> nom = matrix_t<T>::Zero(x, z);
    #pragma omp parallel for
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            for (size_t k = 0; k < z; ++k) {
                nom(i, k) +=
                    orig_over_appr(i, j) * mu(k, j) * grad_plus(i, j, k);
            }
        }
    }
    #pragma omp parallel for
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            for (size_t k = 0; k < z; ++k) {
                for (size_t c = 0; c < z; ++c) {
                    nom(i, k) += orig_over_appr_squared(i, j) * mu(k, j) *
                                 nu(i, c) * mu(c, j) * grad_minus(i, c);
                }
            }
        }
    }
    matrix_t<T> denom = matrix_t<T>::Zero(x, z);
    #pragma omp parallel for
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            for (size_t k = 0; k < z; ++k) {
                denom(i, k) +=
                    orig_over_appr(i, j) * mu(k, j) * grad_minus(i, k);
            }
        }
    }
    #pragma omp parallel for
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            for (size_t k = 0; k < z; ++k) {
                for (size_t c = 0; c < z; ++c) {
                    denom(i, k) += orig_over_appr_squared(i, j) * mu(k, j) *
                                   nu(i, c) * mu(c, j) * grad_plus(i, j, c);
                }
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x; ++i) {
        for (size_t k = 0; k < z; ++k) {
            nu(i, k) *= nom(i, k) / (denom(i, k) + eps);
        }
    }

    // normalize
    vector_t<T> nu_rowsum_plus_eps =
        nu.rowwise().sum() + vector_t<T>::Constant(nu.rows(), eps).transpose();
    nu = nu.array().colwise() / nu_rowsum_plus_eps.transpose().array();
}

template <typename T>
static void update_mu(const matrix_t<T>& orig_over_appr,
                      const matrix_t<T>& orig_over_appr_squared,
                      const matrix_t<T>& nu, const tensor_t<T, 3>& grad_plus,
                      const matrix_t<T>& grad_minus, double eps,
                      matrix_t<T>& mu) {
    auto x = static_cast<size_t>(grad_plus.dimension(0));
    auto y = static_cast<size_t>(grad_plus.dimension(1));
    auto z = static_cast<size_t>(grad_plus.dimension(2));

    matrix_t<T> nom = matrix_t<T>::Zero(z, y);
    #pragma omp parallel for
    for (size_t j = 0; j < y; ++j) {
        for (size_t k = 0; k < z; ++k) {
            for (size_t i = 0; i < x; ++i) {
                nom(k, j) +=
                    orig_over_appr(i, j) * nu(i, k) * grad_plus(i, j, k);
            }
        }
    }
    #pragma omp parallel for
    for (size_t j = 0; j < y; ++j) {
        for (size_t k = 0; k < z; ++k) {
            for (size_t i = 0; i < x; ++i) {
                for (size_t c = 0; c < z; ++c) {
                    nom(k, j) += orig_over_appr_squared(i, j) * nu(i, k) *
                                 nu(i, c) * mu(c, j) * grad_minus(i, c);
                }
            }
        }
    }
    matrix_t<T> denom = matrix_t<T>::Zero(z, y);
    #pragma omp parallel for
    for (size_t j = 0; j < y; ++j) {
        for (size_t k = 0; k < z; ++k) {
            for (size_t i = 0; i < x; ++i) {
                denom(k, j) +=
                    orig_over_appr(i, j) * nu(i, k) * grad_minus(i, k);
            }
        }
    }
    #pragma omp parallel for
    for (size_t j = 0; j < y; ++j) {
        for (size_t k = 0; k < z; ++k) {
            for (size_t i = 0; i < x; ++i) {
                for (size_t c = 0; c < z; ++c) {
                    denom(k, j) += orig_over_appr_squared(i, j) * nu(i, k) *
                                   nu(i, c) * mu(c, j) * grad_plus(i, j, c);
                }
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < z; ++k) {
        for (size_t j = 0; j < y; ++j) {
            mu(k, j) *= nom(k, j) / (denom(k, j) + eps);
        }
    }

    // normalize
    vector_t<T> mu_colsum_plus_eps =
        mu.colwise().sum() + vector_t<T>::Constant(mu.cols(), eps);
    mu = mu.array().rowwise() / mu_colsum_plus_eps.array();
}
} // namespace details

namespace bld {
/**
 * @brief Compute tensor \f$S\f$, the solution of BLD problem \cite
 * kurutmazbayesian, and optimization matrices \f$\mu\f$ and \f$\nu\f$ from
 * matrix \f$X\f$ using the approximate multiplicative algorithm given in
 * \cite kurutmazbayesian.
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
 * This algorithm finds the optimal \f$S\f$ using the multiplicative algorithm
 * employing Lagrange multipliers as given in \cite kurutmazbayesian:
 *
 * \f[
 * S_{ijk} = X_{ij}\frac{\nu_{ik}\mu_{kj}}{\sum_c \nu_{ic}\mu{cj}}.
 * \f]
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
 * @return std::tuple of tensor \f$S\f$ of size \f$x \times y \times z\f$ where
 * \f$X = S_{ij+}\f$, and matrices \f$\nu\f$ and \f$\mu\f$.
 *
 * @throws std::invalid_argument if X contains negative entries,
 * if number of rows of X is not equal to number of alpha parameters, if z is
 * not equal to number of beta parameters.
 */
template <typename T, typename Scalar>
std::tuple<tensor_t<T, 3>, matrix_t<T>, matrix_t<T>>
bld_appr(const matrix_t<T>& X, size_t z,
         const alloc_model::Params<Scalar>& model_params,
         size_t max_iter = 1000, double eps = 1e-50) {
    details::check_bld_params(X, z, model_params);

    auto x = static_cast<size_t>(X.rows());
    auto y = static_cast<size_t>(X.cols());

    vector_t<Scalar> alpha(model_params.alpha.size());
    vector_t<Scalar> beta(model_params.beta.size());
    std::copy(model_params.alpha.begin(), model_params.alpha.end(),
              alpha.data());
    std::copy(model_params.beta.begin(), model_params.beta.end(), beta.data());

    matrix_t<T> nu(x, z);
    matrix_t<T> mu(y, z);
    // initialize nu and mu using Dirichlet
    {
        util::gsl_rng_wrapper rnd_gen(gsl_rng_alloc(gsl_rng_taus),
                                      gsl_rng_free);
        std::vector<double> dirichlet_params(z, 1);

        // remark: following code depends on the fact that matrixd is row
        // major
        for (size_t i = 0; i < x; ++i) {
            gsl_ran_dirichlet(rnd_gen.get(), z, dirichlet_params.data(),
                              nu.data() + i * z);
        }
        for (size_t j = 0; j < y; ++j) {
            gsl_ran_dirichlet(rnd_gen.get(), z, dirichlet_params.data(),
                              mu.data() + j * z);
        }

        mu.transposeInPlace();
    }

    tensor_t<T, 3> grad_plus(x, y, z);
    matrix_t<T> grad_minus(x, z);
    tensor_t<T, 3> S(x, y, z);
    matrix_t<T> eps_mat = matrix_t<T>::Constant(x, y, eps);
    matrix_t<T> X_hat, orig_over_appr, orig_over_appr_squared, alpha_eph,
        beta_eph;

    for (size_t eph = 0; eph < max_iter; ++eph) {
        // update helpers
        details::update_X_hat(nu, mu, X_hat);
        details::update_orig_over_appr(X, X_hat, eps_mat,
                                                  orig_over_appr);
        details::update_orig_over_appr_squared(
            X, X_hat, eps_mat, orig_over_appr_squared);
        details::update_S(orig_over_appr, nu, mu, S);
        details::update_alpha_eph(alpha, S, alpha_eph);
        details::update_beta_eph(beta, S, beta_eph);
        details::update_grad_plus(beta_eph, S, grad_plus);
        details::update_grad_minus(alpha_eph, grad_minus);

        // update nu
        details::update_nu(orig_over_appr, orig_over_appr_squared,
                                      mu, grad_plus, grad_minus, eps, nu);

        // update helpers
        details::update_X_hat(nu, mu, X_hat);
        details::update_orig_over_appr(X, X_hat, eps_mat,
                                                  orig_over_appr);
        details::update_orig_over_appr_squared(
            X, X_hat, eps_mat, orig_over_appr_squared);
        details::update_S(orig_over_appr, nu, mu, S);
        details::update_alpha_eph(alpha, S, alpha_eph);
        details::update_beta_eph(beta, S, beta_eph);
        details::update_grad_plus(beta_eph, S, grad_plus);
        details::update_grad_minus(alpha_eph, grad_minus);

        // update mu
        details::update_mu(orig_over_appr, orig_over_appr_squared,
                                      nu, grad_plus, grad_minus, eps, mu);
    }

    details::update_X_hat(nu, mu, X_hat);
    details::update_orig_over_appr(X, X_hat, eps_mat,
                                              orig_over_appr);
    details::update_S(orig_over_appr, nu, mu, S);

    return std::make_tuple(S, nu, mu);
}
} // namespace bld
} // namespace bnmf_algs
