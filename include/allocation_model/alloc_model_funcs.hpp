#pragma once

#include "allocation_model/alloc_model_params.hpp"
#include "defs.hpp"
#include <tuple>
#include <util/generator.hpp>
#include <util/wrappers.hpp>
#include <vector>

namespace bnmf_algs {

/**
 * @brief Namespace that contains functions related to Allocation Model \cite
 * kurutmazbayesian.
 */
namespace allocation_model {
/**
 * @brief Return prior matrices W, H and vector L according to the Bayesian
 * NMF allocation model using distribution parameters.
 *
 * According to the Bayesian NMF allocation model \cite kurutmazbayesian,
 *
 * \f$L_j \sim \mathcal{G}(a, b) \qquad W_{:k} \sim \mathcal{D}(\alpha)
 * \qquad H_{:j} \sim \mathcal{D}(\beta)\f$.
 *
 * @param tensor_shape Shape of the sample tensor \f$x \times y \times z\f$.
 * @param model_params Allocation Model parameters. See
 * bnmf_algs::allocation_model::AllocModelParams.
 *
 * @return std::tuple of \f$<W_{x \times z}, H_{z \times y}, L_y>\f$.
 *
 * @throws std::invalid_argument if number of alpha parameters is not \f$x\f$,
 * if number of beta parameters is not \f$z\f$.
 */
std::tuple<matrixd, matrixd, vectord>
bnmf_priors(const shape<3>& tensor_shape, const AllocModelParams& model_params);

/**
 * @brief Sample a tensor S from generative Bayesian NMF model using the
 * given priors.
 *
 * According to the generative Bayesian NMF model \cite kurutmazbayesian,
 *
 * \f$S_{ijk} \sim \mathcal{PO}(W_{ik}H_{kj}L_{j})\f$.
 *
 * @param prior_W Prior matrix for \f$W\f$ of shape \f$x \times z\f$.
 * @param prior_H Prior matrix for \f$H\f$ of shape \f$z \times y\f$.
 * @param prior_L Prior vector for \f$L\f$ of size \f$y\f$.
 *
 * @return A sample tensor of size \f$x \times y \times z\f$ from
 * generative Bayesian NMF model.
 *
 * @throws std::invalid_argument if number of columns of prior_W is not equal to
 * number of rows of prior_H, if number of columns of prior_H is not equal to
 * number of columns of prior_L
 */
tensord<3> sample_S(const matrixd& prior_W, const matrixd& prior_H,
                    const vectord& prior_L);

/**
 * @brief Compute the log marginal of tensor S with respect to the given
 * distribution parameters.
 *
 * According to the Bayesian NMF allocation model \cite kurutmazbayesian,
 *
 * \f$\log{p(S)} = \log{p(W, H, L, S)} - \log{p(W, H, L | S)}\f$.
 *
 * @param S Tensor \f$S\f$ of size \f$x \times y \times z \f$.
 * @param model_params Allocation model parameters. See
 * bnmf_algs::allocation_model::AllocModelParams.
 *
 * @return log marginal \f$\log{p(S)}\f$.
 *
 * @throws std::invalid_argument if number of alpha parameters is not equal to
 * S.dimension(0), if number of beta parameters is not equal to S.dimension(2).
 */
double log_marginal_S(const tensord<3>& S,
                      const AllocModelParams& model_params);
} // namespace allocation_model
} // namespace bnmf_algs
