#pragma once

#include "alloc_model/alloc_model_params.hpp"
#include "defs.hpp"
#include "util/util.hpp"
#include "util_details.hpp"
#include <gsl/gsl_sf_psi.h>

#ifdef USE_OPENMP
#include <omp.h>
#include <thread>
#endif

#ifdef USE_CUDA
#include "cuda/memory.hpp"
#include "cuda/tensor_ops.hpp"
#endif

namespace bnmf_algs {
namespace bld {
/**
 * @brief Compute tensor \f$S\f$, the solution of BLD problem \cite
 * kurutmazbayesian, from matrix \f$X\f$ using multiplicative update rules.
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
 * Multiplicative BLD algorithm solves a constrained optimization problem over
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
 *     S_{ijk} = \phi(Z_{ijk}) = \frac{Z_{ijk}X_{ij}}{\sum_{k'}Z_{ijk'}}
 * \f]
 *
 * This particular choice of \f$\phi\f$ leads to multiplicative update rules
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
 * @param use_psi_appr If true, use util::psi_appr function to compute the
 * digamma function approximately. If false, compute the digamma function
 * exactly.
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
tensor_t<T, 3> bld_mult(const matrix_t<T>& X, size_t z,
                        const alloc_model::Params<Scalar>& model_params,
                        size_t max_iter = 1000, bool use_psi_appr = false,
                        double eps = 1e-50) {
    details::check_bld_params(X, z, model_params);

#ifdef USE_OPENMP
    const int num_threads = std::thread::hardware_concurrency();
#endif

    auto x = static_cast<size_t>(X.rows());
    auto y = static_cast<size_t>(X.cols());

    shape<3> dims = {x, y, z};

    // initialize tensor S
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

#ifdef USE_CUDA
    cuda::HostMemory2D<const T> X_host(X.data(), x, y);
    cuda::DeviceMemory2D<T> X_device(x, y);
    cuda::copy2D(X_device, X_host);
#endif

    tensor_t<T, 3> grad_plus(x, y, z);
    matrix_t<T> grad_minus(x, z);
    matrix_t<T> nom_mult(x, y);
    matrix_t<T> denom_mult(x, y);
    matrix_t<T> alpha_eph(x, z);
    vector_t<T> alpha_eph_sum(z);
    matrix_t<T> beta_eph(y, z);
    tensor_t<T, 2> S_pjk_tensor(y, z);
    tensor_t<T, 2> S_ipk_tensor(x, z);
    tensor_t<T, 2> S_ijp_tensor(x, y);

#ifdef USE_CUDA
    cuda::HostMemory1D<T> S_host(S.data(), x * y * z);
    std::array<cuda::HostMemory1D<T>, 3> host_sums = {
        cuda::HostMemory1D<T>(S_pjk_tensor.data(), y * z),
        cuda::HostMemory1D<T>(S_ipk_tensor.data(), x * z),
        cuda::HostMemory1D<T>(S_ijp_tensor.data(), x * y)};

    cuda::DeviceMemory1D<T> S_device(x * y * z);
    std::array<cuda::DeviceMemory1D<T>, 3> device_sums = {
        cuda::DeviceMemory1D<T>(y * z), cuda::DeviceMemory1D<T>(x * z),
        cuda::DeviceMemory1D<T>(x * y)};

    for (size_t i = 0; i < 3; ++i) {
        cuda::copy1D(device_sums[i], host_sums[i]);
    }
#endif

    Eigen::Map<const vector_t<Scalar>> alpha(model_params.alpha.data(),
                                             model_params.alpha.size());
    Eigen::Map<const vector_t<Scalar>> beta(model_params.beta.data(),
                                            model_params.beta.size());

#ifdef USE_OPENMP
    Eigen::ThreadPool tp(num_threads);
    Eigen::ThreadPoolDevice thread_dev(&tp, num_threads);
#endif

    const auto psi_fn = use_psi_appr ? util::psi_appr<T> : gsl_sf_psi;
    for (size_t eph = 0; eph < max_iter; ++eph) {
        // update S_pjk, S_ipk, S_ijp
#ifdef USE_CUDA
        cuda::copy1D(S_device, S_host);
        cuda::tensor_sums(S_device, dims, device_sums);

        for (size_t i = 0; i < 3; ++i) {
            cuda::copy1D(host_sums[i], device_sums[i]);
        }
#elif USE_OPENMP
        S_pjk_tensor.device(thread_dev) = S.sum(shape<1>({0}));
        S_ipk_tensor.device(thread_dev) = S.sum(shape<1>({1}));
        S_ijp_tensor.device(thread_dev) = S.sum(shape<1>({2}));
#else
        S_pjk_tensor = S.sum(shape<1>({0}));
        S_ipk_tensor = S.sum(shape<1>({1}));
        S_ijp_tensor = S.sum(shape<1>({2}));
#endif

        Eigen::Map<matrix_t<T>> S_pjk(S_pjk_tensor.data(), y, z);
        Eigen::Map<matrix_t<T>> S_ipk(S_ipk_tensor.data(), x, z);
        Eigen::Map<matrix_t<T>> S_ijp(S_ijp_tensor.data(), x, y);

        // update alpha_eph, beta_eph
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < z; ++k) {
            for (size_t i = 0; i < x; ++i) {
                alpha_eph(i, k) = S_ipk(i, k) + alpha(i);
            }

            for (size_t j = 0; j < y; ++j) {
                beta_eph(j, k) = S_pjk(j, k) + beta(k);
            }
        }

#ifdef USE_CUDA
        //cuda::bld_mult::update_grad_plus(S, beta_eph, grad_plus);
//#else
        // update grad_plus
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < x; ++i) {
            for (size_t j = 0; j < y; ++j) {
                for (size_t k = 0; k < z; ++k) {
                    grad_plus(i, j, k) =
                        psi_fn(beta_eph(j, k)) - psi_fn(S(i, j, k) + 1);
                }
            }
        }
#endif

        // update alpha_eph_sum
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < z; ++k) {
            alpha_eph_sum(k) = 0;
            for (size_t i = 0; i < x; ++i) {
                alpha_eph_sum(k) += alpha_eph(i, k);
            }
        }

        // update grad_minus
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < x; ++i) {
            for (size_t k = 0; k < z; ++k) {
                grad_minus(i, k) =
                    psi_fn(alpha_eph_sum(k)) - psi_fn(alpha_eph(i, k));
            }
        }

        // update nom_mult, denom_mult
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < x; ++i) {
            for (size_t j = 0; j < y; ++j) {
                T xdiv = 1.0 / (X(i, j) + eps);
                T nom_sum = 0, denom_sum = 0;
                for (size_t k = 0; k < z; ++k) {
                    nom_sum += (S(i, j, k) * xdiv * grad_minus(i, k));
                    denom_sum += (S(i, j, k) * xdiv * grad_plus(i, j, k));
                }
                nom_mult(i, j) = nom_sum;
                denom_mult(i, j) = denom_sum;
            }
        }

        // update S
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < x; ++i) {
            for (size_t j = 0; j < y; ++j) {
                T s_ij = S_ijp(i, j);
                T x_over_s = X(i, j) / (s_ij + eps);
                for (size_t k = 0; k < z; ++k) {
                    S(i, j, k) *= (grad_plus(i, j, k) + nom_mult(i, j)) *
                                  x_over_s /
                                  (grad_minus(i, k) + denom_mult(i, j) + eps);
                }
            }
        }
    }

    return S;
}
} // namespace bld
} // namespace bnmf_algs
