#pragma once

#include <Eigen/Dense>
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
std::pair<matrixd, matrixd> nmf(const matrixd& X, size_t r, double beta,
                                  size_t max_iter = 1000);

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
 * @param x First value. If \f$x=0\f$ and beta is 0 (IS) or 1 (KL), the terms
 * containing x is not used during IS or KL computation.
 * @param y Second value.
 * @param beta \f$\beta\f$-divergence parameter.
 * @param eps Epsilon value to prevent division by 0.
 *
 * @return \f$\beta\f$-divergence between the two scalars.
 */
double beta_divergence(double x, double y, double beta, double eps = 1e-50);

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
double beta_divergence(InputIterator1 first_begin, InputIterator1 first_end,
                       InputIterator2 second_begin, double beta,
                       double eps = 1e-50) {
    // todo: parallelize
    return std::inner_product(first_begin, first_end, second_begin, 0.0,
                              std::plus<>(), [beta, eps](double x, double y) {
                                  return beta_divergence(x, y, beta, eps);
                              });
}

/**
 * @brief Return \f$\beta\f$-divergence between two vectors/matrices/tensors
 * as defined in bnmf_algs::vectord, bnmf_algs::matrixd, bnmf_algs::tensord.
 *
 * @tparam Tensor A contiguous data holder type with data() and size() members.
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
double beta_divergence(const Tensor& X, const Tensor& Y, double beta,
                       double eps = 1e-50) {
    const auto size = X.size();

    return beta_divergence(X.data(), X.data() + size, Y.data(), beta, eps);
}
} // namespace nmf
} // namespace bnmf_algs
