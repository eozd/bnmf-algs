#pragma once

#include "alloc_model/alloc_model_params.hpp"
#include "bld/util_details.hpp"
#include "bld_mult_funcs.hpp"
#include "defs.hpp"
#include "util/util.hpp"
#include <gsl/gsl_sf_psi.h>

#ifdef USE_OPENMP
#include <thread>
#endif

#ifdef USE_CUDA
#include "bld_mult_cuda_funcs.hpp"
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
 * @throws assertion error if X contains negative entries,
 * if number of rows of X is not equal to number of alpha parameters, if z is
 * not equal to number of beta parameters.
 */
template <typename T, typename Scalar>
tensor_t<T, 3> bld_mult(const matrix_t<T>& X, const size_t z,
                        const alloc_model::Params<Scalar>& model_params,
                        size_t max_iter = 1000, bool use_psi_appr = false,
                        double eps = 1e-50) {
    details::check_bld_params(X, z, model_params);

    const auto x = static_cast<size_t>(X.rows());
    const auto y = static_cast<size_t>(X.cols());

    // computation variables
    tensor_t<T, 3> S = details::bld_mult::init_S(X, z);
    const matrix_t<T> X_reciprocal = details::bld_mult::X_reciprocal(X, eps);

    tensor_t<T, 3> grad_plus(x, y, z);
    matrix_t<T> nom_mult(x, y);
    matrix_t<T> denom_mult(x, y);
    matrix_t<T> grad_minus(x, z);
    matrix_t<T> alpha_eph(x, z);
    vector_t<T> alpha_eph_sum(z);
    matrix_t<T> beta_eph(y, z);
    tensor_t<T, 2> S_pjk(y, z);
    tensor_t<T, 2> S_ipk(x, z);
    tensor_t<T, 2> S_ijp(x, y);
    vector_t<T> alpha;
    vector_t<T> beta;

    std::tie(alpha, beta) = details::bld_mult::init_alpha_beta(model_params);

    // initialize threads
#ifdef USE_OPENMP
    Eigen::ThreadPool tp(std::thread::hardware_concurrency());
    Eigen::ThreadPoolDevice thread_dev(&tp,
                                       std::thread::hardware_concurrency());
#endif

    // which psi function to use
    const auto psi_fn = use_psi_appr ? util::psi_appr<T> : gsl_sf_psi;

    // iterations
    for (size_t eph = 0; eph < max_iter; ++eph) {
        // update S_pjk, S_ipk, S_ijp
#ifdef USE_OPENMP
        S_pjk.device(thread_dev) = S.sum(shape<1>({0}));
        S_ipk.device(thread_dev) = S.sum(shape<1>({1}));
        S_ijp.device(thread_dev) = S.sum(shape<1>({2}));
#else
        S_pjk_tensor = S.sum(shape<1>({0}));
        S_ipk_tensor = S.sum(shape<1>({1}));
        S_ijp_tensor = S.sum(shape<1>({2}));
#endif

        details::bld_mult::update_alpha_eph(S_ipk, alpha, alpha_eph);
        details::bld_mult::update_beta_eph(S_pjk, beta, beta_eph);

        details::bld_mult::update_grad_plus(S, beta_eph, psi_fn, grad_plus);
        details::bld_mult::update_grad_minus(alpha_eph, psi_fn, grad_minus);

        details::bld_mult::update_nom_mult(X_reciprocal, grad_minus, S,
                                           nom_mult);
        details::bld_mult::update_denom_mult(X_reciprocal, grad_plus, S,
                                             denom_mult);
        details::bld_mult::update_S(X, nom_mult, denom_mult, grad_minus,
                                    grad_plus, S_ijp, S, eps);
    }

    return S;
}

#ifdef USE_CUDA

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
 * This function is provided only when the library is built with CUDA support.
 * Large matrix and tensor updates employed in bld_mult algorithm are performed
 * using CUDA. In particular, updating the tensor S, its gradient tensors and
 * certain highly parallelized \f$O(xyz)\f$ operations are performed on the GPU
 * to speed up the computation. Tensors and matrices are copied to/from the GPU
 * only when needed. Additionally, objects that are only used during GPU updates
 * are never allocated on the main memory; they reside only on the GPU for
 * higher efficiency.
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
 * exactly. Note that updates on the GPU use
 * bnmf_algs::details::kernel::psi_appr function since they can't use any third
 * party library functions.
 * @param eps Floating point epsilon value to be used to prevent division by 0
 * errors.
 *
 * @return Tensor \f$S\f$ of size \f$x \times y \times z\f$ where \f$X =
 * S_{ij+}\f$.
 *
 * @throws assertion error if X contains negative entries,
 * if number of rows of X is not equal to number of alpha parameters, if z is
 * not equal to number of beta parameters.
 */
template <typename T, typename Scalar>
tensor_t<T, 3> bld_mult_cuda(const matrix_t<T>& X, const size_t z,
                             const alloc_model::Params<Scalar>& model_params,
                             size_t max_iter = 1000, bool use_psi_appr = false,
                             double eps = 1e-50) {
    details::check_bld_params(X, z, model_params);

    const auto x = static_cast<size_t>(X.rows());
    const auto y = static_cast<size_t>(X.cols());

    // initial S
    tensor_t<T, 3> S = details::bld_mult::init_S(X, z);

    // X_reciprocal(i, j) = 1 / X(i, j)
    const matrix_t<T> X_reciprocal = details::bld_mult::X_reciprocal(X, eps);

    matrix_t<T> grad_minus(x, z);
    matrix_t<T> alpha_eph(x, z);
    vector_t<T> alpha_eph_sum(z);
    matrix_t<T> beta_eph(y, z);
    tensor_t<T, 2> S_pjk(y, z);
    tensor_t<T, 2> S_ipk(x, z);
    tensor_t<T, 2> S_ijp(x, y);
    vector_t<T> alpha;
    vector_t<T> beta;

    std::tie(alpha, beta) = details::bld_mult::init_alpha_beta(model_params);

    // host memory wrappers to be used during copying between main memory and
    // GPU
    cuda::HostMemory3D<T> S_host(S.data(), x, y, z);
    cuda::HostMemory2D<T> S_pjk_host(S_pjk.data(), y, z);
    cuda::HostMemory2D<T> S_ipk_host(S_ipk.data(), x, z);
    cuda::HostMemory2D<T> beta_eph_host(beta_eph.data(), y, z);
    cuda::HostMemory2D<T> grad_minus_host(grad_minus.data(), x, z);

    // allocate memory on GPU
    cuda::DeviceMemory2D<T> X_device(x, y);
    cuda::DeviceMemory2D<T> X_reciprocal_device(x, y);
    cuda::DeviceMemory3D<T> S_device(x, y, z);
    std::array<cuda::DeviceMemory2D<T>, 3> device_sums = {
        cuda::DeviceMemory2D<T>(y, z), cuda::DeviceMemory2D<T>(x, z),
        cuda::DeviceMemory2D<T>(x, y)};
    cuda::DeviceMemory3D<T> grad_plus_device(x, y, z);
    cuda::DeviceMemory2D<T> beta_eph_device(y, z);
    cuda::DeviceMemory2D<T> nom_device(x, y);
    cuda::DeviceMemory2D<T> denom_device(x, y);
    cuda::DeviceMemory2D<T> grad_minus_device(x, z);

    // send initial variables to GPU
    cuda::copy3D(S_device, S_host);
    {
        // X and X_reciprocal will be sent only once to device
        cuda::HostMemory2D<const T> X_host(X.data(), x, y);
        cuda::HostMemory2D<const T> X_reciprocal_host(X_reciprocal.data(), x,
                                                      y);
        cuda::copy2D(X_device, X_host);
        cuda::copy2D(X_reciprocal_device, X_reciprocal_host);
    }

    // which psi function to use
    const auto psi_fn = use_psi_appr ? util::psi_appr<T> : gsl_sf_psi;

    // iterations
    for (size_t eph = 0; eph < max_iter; ++eph) {
        // update S_pjk, S_ipk, S_ijp
        cuda::tensor_sums(S_device, device_sums);
        cuda::copy2D(S_pjk_host, device_sums[0]);
        cuda::copy2D(S_ipk_host, device_sums[1]);

        details::bld_mult::update_beta_eph(S_pjk, beta, beta_eph);

        // update grad_plus using CUDA
        cuda::copy2D(beta_eph_device, beta_eph_host);
        details::bld_mult::update_grad_plus_cuda(S_device, beta_eph_device,
                                                 grad_plus_device);
        // update denom using CUDA
        details::bld_mult::update_denom_cuda(
                X_reciprocal_device, grad_plus_device, S_device, denom_device);

        // above two cuda calls are asynchronous; immediately start working on
        // the CPU

        details::bld_mult::update_alpha_eph(S_ipk, alpha, alpha_eph);
        details::bld_mult::update_grad_minus(alpha_eph, psi_fn, grad_minus);

        // copy synchronizes
        cuda::copy2D(grad_minus_device, grad_minus_host);
        details::bld_mult::update_nom_cuda(
            X_reciprocal_device, grad_minus_device, S_device, nom_device);

        // update S using CUDA
        const auto& S_ijp_device = device_sums[2];
        details::bld_mult::update_S_cuda(X_device, nom_device, denom_device,
                                         grad_minus_device, grad_plus_device,
                                         S_ijp_device, S_device);
    }
    cuda::copy3D(S_host, S_device);
    return S;
}

#endif
} // namespace bld
} // namespace bnmf_algs
