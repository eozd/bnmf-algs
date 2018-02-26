#pragma once

#include <vector>
#include "defs.hpp"

namespace bnmf_algs {

struct AllocModelParams {
    /**
     * @brief Shape parameter of Gamma distribution.
     */
    double a;
    /**
     * @brief Rate parameter of Gamma distribution.
     */
    double b;
    /**
     * @brief Parameter vector of Dirichlet prior for matrix \f$W_{x \times
     * y}\f$ of size \f$x\f$.
     */
    std::vector<double> alpha;
    /**
     * @brief Parameter vector of Dirichlet prior for matrix \f$H_{z \times
     * y}\f$ of size \f$z\f$.
     */
    std::vector<double> beta;

    /**
     * @brief Construct AllocModelParams manually with the given distribution
     * parameters.
     *
     * @param a Shape parameter of Gamma distribution.
     * @param b Rate parameter of Gamma distribution.
     * @param alpha Parameter vector of Dirichlet prior for matrix \f$W_{x
     * \times y}\f$ of size \f$x\f$.
     * @param beta Parameter vector of Dirichlet prior for matrix \f$H_{z \times
     * y}\f$ of size \f$z\f$.
     */
    AllocModelParams(double a, double b, const std::vector<double>& alpha,
                     const std::vector<double>& beta);

    /**
     * @brief Construct AllocModelParams with the default distribution
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
    explicit AllocModelParams(const shape<3>& tensor_shape);
};
} // namespace bnmf_algs
