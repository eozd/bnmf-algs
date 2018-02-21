#pragma once

#include <vector>
#include "defs.hpp"

namespace bnmf_algs {
    /**
     * @brief Sample a tensor S from Bayesian NMF generative model using
     * distribution parameters.
     *
     * According to the generative Bayesian NMF model \cite kurutmazbayesian,
     *
     * \f$L_j \sim \mathcal{G}(a, b) \qquad W_{:k} \sim \mathcal{D}(\alpha)
     * \qquad H_{:j} \sim \mathcal{D}(\beta)\f$.
     *
     * @param tensor_shape Shape of the sample tensor \f$x \times y \times z\f$.
     * @param a Shape parameter of Gamma distribution.
     * @param b Rate parameter of Gamma distribution.
     * @param alpha Parameter vector of Dirichlet prior for matrix \f$W\f$ of size \f$x\f$.
     * @param beta Parameter vector of Dirichlet prior for matrix \f$H\f$ of size \f$z\f$.
     *
     * @return A sample tensor from the generative model of shape \f$x \times y \times z\f$.
     *
     * @remark Number of alpha parameters must be \f$x\f$.
     * @remark Number of beta parameters must be \f$z\f$.
     *
     * @author Esref Ozdemir
     */
    tensor3d_t sample_S(const std::tuple<int, int, int>& tensor_shape,
                        double a,
                        double b,
                        const std::vector<double>& alpha,
                        const std::vector<double>& beta);

    /**
     * @brief Sample a tensor S from generative Bayesian NMF model using the
     * given priors.
     *
     * According to the generative Bayesian NMF model \cite kurutmazbayesian,
     *
     * \f$S_{ijk} \sim \mathcal{PO}(W_{ik}H_{kj}L_{j})\f$.
     *
     * @param prior_L Prior vector for \f$L\f$ of size \f$y\f$.
     * @param prior_W Prior matrix for \f$W\f$ of shape \f$x \times z\f$.
     * @param prior_H Prior matrix for \f$H\f$ of shape \f$z \times y\f$.
     *
     * @return A sample tensor of size \f$x \times y \times z\f$ from
     * generative Bayesian NMF model.
     *
     * @author Esref Ozdemir
     */
    tensor3d_t sample_S(const vector_t& prior_L,
                        const matrix_t& prior_W,
                        const matrix_t& prior_H);
}