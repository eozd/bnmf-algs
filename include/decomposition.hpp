#pragma once

#include "alloc_model_params.hpp"
#include "defs.hpp"

namespace bnmf_algs {
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
 * \todo Explain sequential greedy algorithm. (first need to understand)
 *
 * @param X Nonnegative matrix of size \f$x \times y\f$ to decompose.
 * @param z Number of matrices into which matrix \f$X\f$ will be decomposed.
 * This is the depth of the output tensor \f$S\f$.
 * @param alpha Parameter vector of Dirichlet prior for matrix \f$W\f$ of size
 * \f$x\f$.
 * @param beta Parameter vector of Dirichlet prior for matrix \f$H\f$ of size
 * \f$z\f$.
 *
 * @return Tensor \f$S\f$ of size \f$x \times y \times z\f$ where \f$X =
 * S_{ij+}\f$.
 *
 * @throws std::invalid_argument if matrix \f$X\f$ is not nonnegative, if size
 * of alpha is not \f$x\f$, if size of beta is not \f$z\f$.
 */
tensor3d_t seq_greedy_bld(const matrix_t& X, size_t z,
                          const AllocModelParams& model_params);

/**
 * @brief Compute matrices \f$W, H\f$ and vector \f$L\f$ from tensor \f$S\f$
 * according to the allocation model.
 *
 *
 * According to Allocation Model \cite kurutmazbayesian,
 *
 * \f[
 * L_j \sim \mathcal{G}(a, b) \qquad W_{:k} \sim \mathcal{D}(\alpha) \qquad
 * H_{:j} \sim \mathcal{D}(\beta) \f]
 *
 * Each entry \f$S_{ijk} \sim \mathcal{PO}(W_{ik}H_{kj}L_j)\f$.
 *
 * @param S Tensor \f$S\f$ of shape \f$x \times y \times z\f$.
 * @param model_params Allocation model parameters. See
 * bnmf_algs::AllocModelParams.
 * @param epsilon Floating point epsilon value to be used during normalization.
 *
 * @return std::tuple of matrices \f$W_{x \times z}, H_{z \times y}\f$ and
 * vector \f$L_y\f$.
 */
std::tuple<matrix_t, matrix_t, vector_t>
bld_fact(const tensor3d_t& S, const AllocModelParams& model_params,
         double epsilon = std::numeric_limits<double>::epsilon());
} // namespace bnmf_algs
