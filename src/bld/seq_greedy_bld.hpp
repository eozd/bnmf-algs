#pragma once

#include "defs.hpp"
#include "util/sampling.hpp"
#include "util_details.hpp"

namespace bnmf_algs {
namespace bld {
/**
 * @brief Compute tensor \f$S\f$, the solution of BLD problem \cite
 * kurutmazbayesian, from matrix \f$X\f$ using sequential greedy method.
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
 * Sequential greedy BLD algorithm works by greedily allocating each entry of
 * input matrix, \f$X_{ij}\f$, one by one to its corresponding bin
 * \f$S_{ijk}\f$. \f$k\f$ is chosen by considering the increase in log
 * marginal value of tensor \f$S\f$ when \f$S_{ijk}\f$ is increased by \f$1\f$.
 * At a given time, \f$i, j\f$ is chosen as the indices of the largest entry of
 * matrix \f$X\f$.
 *
 * @tparam T Type of the matrix/tensor entries.
 * @tparam Scalar Type of the scalars used as model parameters.
 * @param X Nonnegative matrix of size \f$x \times y\f$ to decompose.
 * @param z Number of matrices into which matrix \f$X\f$ will be decomposed.
 * This is the depth of the output tensor \f$S\f$.
 * @param model_params Allocation model parameters. See
 * bnmf_algs::alloc_model::Params<double>.
 *
 * @return Tensor \f$S\f$ of size \f$x \times y \times z\f$ where \f$X =
 * S_{ij+}\f$.
 *
 * @throws std::invalid_argument if matrix \f$X\f$ is not nonnegative, if size
 * of alpha is not \f$x\f$, if size of beta is not \f$z\f$.
 *
 * @remark Worst-case time complexity is \f$O(xy +
 * z(\log{xy})\sum_{ij}X_{ij})\f$ for a matrix \f$X_{x \times y}\f$ and output
 * tensor \f$S_{x \times y \times z}\f$.
 */
template <typename T, typename Scalar>
tensor_t<T, 3> seq_greedy_bld(const matrix_t<T>& X, size_t z,
                              const alloc_model::Params<Scalar>& model_params) {
    details::check_bld_params(X, z, model_params);

    long x = X.rows();
    long y = X.cols();

    // exit early if sum is 0
    if (std::abs(X.sum()) <= std::numeric_limits<double>::epsilon()) {
        tensor_t<T, 3> S(x, y, z);
        S.setZero();
        return S;
    }

    // variables to update during the iterations
    tensor_t<T, 3> S(x, y, z);
    S.setZero();
    matrix_t<T> S_ipk = matrix_t<T>::Zero(x, z); // S_{i+k}
    vector_t<T> S_ppk = vector_t<T>::Zero(z);    // S_{++k}
    matrix_t<T> S_pjk = matrix_t<T>::Zero(y, z); // S_{+jk}
    const Scalar sum_alpha = std::accumulate(model_params.alpha.begin(),
                                             model_params.alpha.end(), 0.0);

    // sample X matrix one by one
    auto matrix_sampler = util::sample_ones_noreplace(X);
    int ii, jj;
    for (const auto& sample : matrix_sampler) {
        std::tie(ii, jj) = sample;

        // find the bin that increases log marginal the largest when it is
        // incremented
        size_t kmax = 0;
        T curr_max = std::numeric_limits<double>::lowest();
        for (size_t k = 0; k < z; ++k) {
            T log_marginal_change =
                std::log(model_params.alpha[ii] + S_ipk(ii, k)) -
                std::log(sum_alpha + S_ppk(k)) - std::log(1 + S(ii, jj, k)) +
                std::log(model_params.beta[k] + S_pjk(jj, k));

            if (log_marginal_change > curr_max) {
                curr_max = log_marginal_change;
                kmax = k;
            }
        }

        // increase the corresponding bin accumulators
        ++S(ii, jj, kmax);
        ++S_ipk(ii, kmax);
        ++S_ppk(kmax);
        ++S_pjk(jj, kmax);
    }

    return S;
}
} // namespace bld
} // namespace bnmf_algs
