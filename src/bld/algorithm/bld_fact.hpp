#pragma once

#include "alloc_model/alloc_model_params.hpp"
#include "defs.hpp"

namespace bnmf_algs {

/**
 * @brief Namespace that contains solver and auxiliary functions for Best Latent
 * Decomposition (BLD) problem.
 */
namespace bld {
/**
 * @brief Compute matrices \f$W, H\f$ and vector \f$L\f$ from tensor \f$S\f$
 * according to the allocation model.
 *
 * According to Allocation Model \cite kurutmazbayesian,
 *
 * \f[
 * L_j \sim \mathcal{G}(a, b) \qquad W_{:k} \sim \mathcal{D}(\alpha) \qquad
 * H_{:j} \sim \mathcal{D}(\beta) \f]
 *
 * Each entry \f$S_{ijk} \sim \mathcal{PO}(W_{ik}H_{kj}L_j)\f$.
 *
 * @tparam T Type of the matrix/tensor entries.
 * @tparam Scalar Type of the scalars used as model parameters.
 * @param S Tensor \f$S\f$ of shape \f$x \times y \times z\f$.
 * @param model_params Allocation model parameters. See
 * bnmf_algs::alloc_model::Params<double>.
 * @param eps Floating point epsilon value to be used to prevent division by 0
 * errors.
 *
 * @return std::tuple of matrices \f$W_{x \times z}, H_{z \times y}\f$ and
 * vector \f$L_y\f$.
 */
template <typename T, typename Scalar>
std::tuple<matrix_t<T>, matrix_t<T>, vector_t<T>>
bld_fact(const tensor_t<T, 3>& S,
         const alloc_model::Params<Scalar>& model_params, double eps = 1e-50) {
    auto x = static_cast<size_t>(S.dimension(0));
    auto y = static_cast<size_t>(S.dimension(1));
    auto z = static_cast<size_t>(S.dimension(2));

    BNMF_ASSERT(model_params.alpha.size() == x,
                "Number of alpha parameters must be equal to S.dimension(0)");
    BNMF_ASSERT(model_params.beta.size() == z,
                "Number of beta parameters must be equal to z");

    tensor_t<T, 2> S_ipk = S.sum(shape<1>({1}));
    tensor_t<T, 2> S_pjk = S.sum(shape<1>({0}));
    tensor_t<T, 1> S_pjp = S.sum(shape<2>({0, 2}));

    matrix_t<T> W(x, z);
    matrix_t<T> H(z, y);
    vector_t<T> L(y);

    for (size_t i = 0; i < x; ++i) {
        for (size_t k = 0; k < z; ++k) {
            W(i, k) = model_params.alpha[i] + S_ipk(i, k) - 1;
        }
    }
    for (size_t k = 0; k < z; ++k) {
        for (size_t j = 0; j < y; ++j) {
            H(k, j) = model_params.beta[k] + S_pjk(j, k) - 1;
        }
    }
    for (size_t j = 0; j < y; ++j) {
        L(j) = (model_params.a + S_pjp(j) - 1) / (model_params.b + 1 + eps);
    }

    // normalize
    vector_t<T> W_colsum =
        W.colwise().sum() + vector_t<T>::Constant(W.cols(), eps);
    vector_t<T> H_colsum =
        H.colwise().sum() + vector_t<T>::Constant(H.cols(), eps);
    W = W.array().rowwise() / W_colsum.array();
    H = H.array().rowwise() / H_colsum.array();

    return std::make_tuple(W, H, L);
}
} // namespace bld
} // namespace bnmf_algs
