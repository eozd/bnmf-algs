#pragma once

#include "defs.hpp"
#include <vector>

namespace bnmf_algs {
namespace alloc_model {

/**
 * @brief Structure to hold the parameters for the Allocation Model \cite
 * kurutmazbayesian.
 *
 * According to the Allocation Model,
 *
 * \f[
 * L_j \sim \mathcal{G}(a, b) \qquad W_{:k} \sim \mathcal{D}(\alpha)
 * \qquad H_{:j} \sim \mathcal{D}(\beta)
 * \f]
 *
 * and \f$S_{ijk} \sim \mathcal{PO}(W_{ik}H_{kj}L_j)\f$.
 *
 * @tparam Scalar Type of the parameters. Common choices are double and float,
 * but integer-like types can also be used.
 */
template <typename Scalar> struct Params {
    /**
     * @brief Shape parameter of Gamma distribution.
     */
    Scalar a;
    /**
     * @brief Rate parameter of Gamma distribution.
     */
    Scalar b;
    /**
     * @brief Parameter vector of Dirichlet prior for matrix \f$W_{x \times
     * y}\f$ of size \f$x\f$.
     */
    std::vector<Scalar> alpha;
    /**
     * @brief Parameter vector of Dirichlet prior for matrix \f$H_{z \times
     * y}\f$ of size \f$z\f$.
     */
    std::vector<Scalar> beta;

    /**
     * @brief Construct Params manually with the given distribution
     * parameters.
     *
     * @param a Shape parameter of Gamma distribution.
     * @param b Rate parameter of Gamma distribution.
     * @param alpha Parameter vector of Dirichlet prior for matrix \f$W_{x
     * \times y}\f$ of size \f$x\f$.
     * @param beta Parameter vector of Dirichlet prior for matrix \f$H_{z \times
     * y}\f$ of size \f$z\f$.
     */
    Params(Scalar a, Scalar b, const std::vector<Scalar>& alpha,
           const std::vector<Scalar>& beta)
        : a(a), b(b), alpha(alpha), beta(beta) {}

    /**
     * @brief Construct Params with the default distribution
     * parameters according to the given tensor shape.
     *
     * Default distribution parameters are
     *
     * \f{eqnarray*}{
     *     \mbox{a} &=& 1 \\
     *     \mbox{b} &=& 10 \\
     *     \mbox{alpha} &=& \begin{matrix}
     *                      (1 & 1 & \dots & 1)
     *                 \end{matrix} \\
     *     \mbox{beta} &=&  \begin{matrix}
     *                      (1 & 1 & \dots & 1)
     *                 \end{matrix}
     * \f}
     *
     * @param tensor_shape Shape of tensor \f$S_{x \times y \times z}\f$.
     */
    explicit Params(const shape<3>& tensor_shape)
        : a(static_cast<Scalar>(1)), b(static_cast<Scalar>(10)),
          alpha(tensor_shape[0], static_cast<Scalar>(1)),
          beta(tensor_shape[2], static_cast<Scalar>(1)) {}

    Params() = default;
};

/**
 * @brief Make default BLD solver parameters.
 *
 * This function makes default BLD parameters as specified in Params::Params.
 *
 * @tparam Scalar Type of the parameters.
 * @param tensor_shape Shape of tensor \f$S_{x \times y \times z}\f$.
 *
 * @return Default parameters to be used with any BLD algorithm.
 */
template <typename Scalar>
Params<Scalar> make_bld_params(const shape<3>& tensor_shape) {
    return Params<Scalar>(tensor_shape);
}

/**
 * @brief Make default EM solver parameters.
 *
 * This function makes default EM algorithm (such as online_EM) parameters. Note
 * that EM algorithms have a different parametrization than BLD solvers, i.e.
 * there is a parameter for each entry of \f$W\f$ and \f$H\f$ matrices.
 * Therefore, this function returns a vector of \f$z\f$ Params objects where
 * \f$z\f$ is the rank of the decomposition (depth of the hidden tensor
 * \f$S\f$.)
 *
 * @tparam Scalar Type of the parameters.
 * @param tensor_shape Shape of tensor \f$S_{x \times y \times z}\f$.
 *
 * @return Default parameters to be used with EM algorithms.
 */
template <typename Scalar>
std::vector<Params<Scalar>> make_EM_params(const shape<3>& tensor_shape) {
    Params<Scalar> bld_params(tensor_shape);
    bld_params.beta =
        std::vector<Scalar>(tensor_shape[1], static_cast<Scalar>(1));

    return std::vector<Params<Scalar>>(tensor_shape[2], bld_params);
}

} // namespace alloc_model
} // namespace bnmf_algs
