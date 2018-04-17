#pragma once

#include <cmath>
#include <limits>
#include <utility>

#include "defs.hpp"

/**
 * @brief Main namespace for bnmf-algs library
 */
namespace bnmf_algs {
/**
 * @brief Namespace that contains solver and auxiliary functions for computing
 * Nonnegative Matrix Factorization (NMF) of a matrix..
 */
namespace nmf {
/**
 * @brief Compute nonnegative matrix factorization of a matrix.
 *
 * According to the NMF model, nonnegative matrix X is factorized into two
 * nonnegative matrices W and H such that
 *
 *          X = WH
 *
 * where X is (m x n), W is (m x r) and H is (r x n) nonnegative matrices. For
 * original NMF paper, see \cite lee-seung-nmf and \cite lee-seung-algs.
 *
 * @tparam T Type of the matrix entries.
 * @tparam Real Type of beta.
 * @param X (m x n) nonnegative matrix.
 * @param r Inner dimension of the factorization. Must be positive.
 * @param beta \f$\beta\f$-divergence parameter. \f$\beta = 0\f$ corresponds to
 * Itakuro-Saito divergence, \f$\beta = 1\f$ corresponds to Kullback-Leibler
 * divergence and \f$\beta = 2\f$ corresponds to Euclidean cost (square of
 * Euclidean distance). Note that beta values different than these values
 * may result in considerably slower convergence.
 * @param max_iter Maximum number of iterations. If set to 0, the algorithm
 *                  runs until convergence.
 *
 * @return std::pair of W and H matrices.
 *
 * @todo Add SVD initialization. See \cite boutsidis2008svd.
 *
 * @throws std::invalid_argument if \f$X\f$ is not nonnegative, if r is not
 * positive, epsilon is not positive
 */
template <typename T, typename Real>
std::pair<matrix_t<T>, matrix_t<T>> nmf(const matrix_t<T>& X, size_t r,
                                        Real beta, size_t max_iter = 1000) {
    const auto x = static_cast<size_t>(X.rows());
    const auto y = static_cast<size_t>(X.cols());
    const auto z = r;

    BNMF_ASSERT((X.array() >= 0).all(), "X matrix must be nonnegative");
    BNMF_ASSERT(r > 0, "r must be positive");

    if (X.isZero(0)) {
        return std::make_pair(matrixd::Zero(x, z), matrixd::Zero(z, y));
    }

    matrix_t<T> W = matrix_t<T>::Random(x, z) + matrix_t<T>::Ones(x, z);
    matrix_t<T> H = matrix_t<T>::Random(z, y) + matrix_t<T>::Ones(z, y);

    if (max_iter == 0) {
        return std::make_pair(W, H);
    }

    const Real p = 2 - beta;
    const double eps = std::numeric_limits<double>::epsilon();
    const auto p_int = static_cast<int>(p);
    const bool p_is_int = std::abs(p_int - p) <= eps;

    matrix_t<T> X_hat, nom_W(x, z), denom_W(x, z), nom_H(z, y), denom_H(z, y);
    while (max_iter-- > 0) {
        // update approximation
        X_hat = W * H;

        // update W
        if (p_is_int && p == 0) {
            // Euclidean NMF
            nom_W = X * H.transpose();
            denom_W = X_hat * H.transpose();
        } else if (p_is_int && p == 1) {
            // KL NMF
            nom_W = (X.array() / X_hat.array()).matrix() * H.transpose();
            denom_W = matrixd::Ones(x, y) * H.transpose();
        } else if (p_is_int && p == 2) {
            // Itakuro-Saito NMF
            nom_W =
                (X.array() / X_hat.array().square()).matrix() * H.transpose();
            denom_W = (X_hat.array().inverse()).matrix() * H.transpose();
        } else {
            // general case
            nom_W = (X.array() / X_hat.array().pow(p)).matrix() * H.transpose();
            denom_W = (X_hat.array().pow(1 - p)).matrix() * H.transpose();
        }
        W = W.array() * nom_W.array() / denom_W.array();

        // update approximation
        X_hat = W * H;

        // update H
        if (p_is_int && p == 0) {
            // Euclidean NMF
            nom_H = W.transpose() * X;
            denom_H = W.transpose() * X_hat;
        } else if (p_is_int && p == 1) {
            // KL NMF
            nom_H = W.transpose() * (X.array() / X_hat.array()).matrix();
            denom_H = W.transpose() * matrixd::Ones(x, y);
        } else if (p_is_int && p == 2) {
            // Itakuro-Saito NMF
            nom_H =
                W.transpose() * (X.array() / X_hat.array().square()).matrix();
            denom_H = W.transpose() * (X_hat.array().inverse()).matrix();
        } else {
            // general case
            nom_H = W.transpose() * (X.array() / X_hat.array().pow(p)).matrix();
            denom_H = W.transpose() * (X_hat.array().pow(1 - p)).matrix();
        }
        H = H.array() * nom_H.array() / denom_H.array();
    }
    return std::make_pair(W, H);
}

/**
 * @brief Compute the \f$\beta\f$-divergence as defined in \cite fevotte2011.
 *
 * This function computes the \f$\beta\f$-divergence between two scalars which
 * is given as
 *
 * \f[
 * d_{\beta}(x|y) =
 * \begin{cases}
 *   \frac{1}{\beta(\beta - 1)}(x^{\beta} + (\beta - 1)y^{\beta} - \beta
 * xy^{\beta - 1}), & \beta \in \mathcal{R}\setminus \{0,1\} \\
 *   x\log{\frac{x}{y}} - x + y, & \beta = 1 \\
 *   \frac{x}{y} - \log{\frac{x}{y}} - 1, & \beta = 0
 * \end{cases}
 * \f]
 *
 * Regular cost functions in NMF can be obtained using certain values of
 * \f$\beta\f$. In particular, \f$\beta = 0\f$ corresponds to Itakuro-Saito
 * divergence, \f$\beta = 1\f$ corresponds to Kullback-Leibler divergence and
 * \f$\beta = 2\f$ corresponds to Euclidean cost (square of Euclidean distance).
 *
 * @tparam Real Type of the values whose beta divergence will be computed.
 * @param x First value. If \f$x=0\f$ and beta is 0 (IS) or 1 (KL), the terms
 * containing x is not used during IS or KL computation.
 * @param y Second value.
 * @param beta \f$\beta\f$-divergence parameter.
 * @param eps Epsilon value to prevent division by 0.
 *
 * @return \f$\beta\f$-divergence between the two scalars.
 */
template <typename Real>
Real beta_divergence(Real x, Real y, Real beta, double eps = 1e-50) {
    if (std::abs(beta) <= std::numeric_limits<double>::epsilon()) {
        // Itakuro Saito
        if (std::abs(x) <= std::numeric_limits<double>::epsilon()) {
            return -1;
        }
        return x / (y + eps) - std::log(x / (y + eps)) - 1;
    } else if (std::abs(beta - 1) <= std::numeric_limits<double>::epsilon()) {
        // KL
        Real logpart;
        if (std::abs(x) <= std::numeric_limits<double>::epsilon()) {
            logpart = 0;
        } else {
            logpart = x * std::log(x / (y + eps));
        }
        return logpart - x + y;
    }

    // general case
    Real nom = std::pow(x, beta) + (beta - 1) * std::pow(y, beta) -
               beta * x * std::pow(y, beta - 1);
    // beta != 0 && beta != 1
    return nom / (beta * (beta - 1));
}

/**
 * @brief Return \f$\beta\f$-divergence between two tensor-like objects as
 * defined in bnmf_algs::vector_t , bnmf_algs::matrix_t , bnmf_algs::tensor_t.
 *
 * @tparam Tensor Type of the tensor-like object with a compile-time
 * <b>value_type</b> field denoting the type of the entries.
 *
 * @param X First tensor-like object (vector/matrix/tensor).
 * @param Y Second tensor-like object (vector/matrix/tensor).
 * @param beta \f$\beta\f$-divergence parameter.
 * @param eps Epsilon value to prevent division by 0.
 *
 * @return Sum of \f$\beta\f$-divergence of each pair of corresponding elements
 * in the given two tensor-like objects.
 *
 * @see nmf::beta_divergence.
 */
template <typename Tensor>
typename Tensor::value_type beta_divergence(const Tensor& X, const Tensor& Y,
                                            typename Tensor::value_type beta,
                                            double eps = 1e-50) {
    typename Tensor::value_type result = 0;
    #pragma omp parallel for schedule(static) reduction(+:result)
    for (long i = 0; i < X.rows(); ++i) {
        for (long j = 0; j < X.cols(); ++j) {
            result += beta_divergence(X(i, j), Y(i, j), beta, eps);
        }
    }

    return result;
}

/**
 * @brief Compute the \f$\beta\f$-divergence as defined in \cite fevotte2011.
 *
 * This function computes the \f$\beta\f$-divergence between each corresponding
 * item in the given two sequences and then returns the summed divergence value.
 *
 * @tparam InputIterator1 Iterator type of the first sequence.
 * @tparam InputIterator2 Iterator type of the second sequence.
 * @param first_begin Beginning of the first sequence.
 * @param first_end End of the first sequence.
 * @param second_begin Beginning of the second sequence.
 * @param beta \f$\beta\f$-divergence parameter.
 * @param eps Epsilon value to prevent division by 0.
 *
 * @return Sum of \f$\beta\f$-divergence of each corresponding element in the
 * given two sequences.
 *
 * @see nmf::beta_divergence.
 */
template <typename InputIterator1, typename InputIterator2>
auto beta_divergence_seq(InputIterator1 first_begin, InputIterator1 first_end,
                         InputIterator2 second_begin,
                         decltype(*first_begin) beta, double eps = 1e-50) {
    // todo: parallelize
    return std::inner_product(first_begin, first_end, second_begin, 0.0,
                              std::plus<>(), [beta, eps](double x, double y) {
                                  return beta_divergence(x, y, beta, eps);
                              });
}
} // namespace nmf
} // namespace bnmf_algs
