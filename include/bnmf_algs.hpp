#pragma once

#include <utility>
#include <Eigen/Dense>

/**
 * bnmf_algs namespace
 */
namespace bnmf_algs {

    /**
     * Nonnegative matrix factorization variant.
     */
    enum class NMFVariant {
        KL,       ///< NMF using Kullback-Leibler divergence as error measure.
                  ///< See \cite lee-seung-algs.

        Euclidean ///< NMF using Euclidean distance as error measure.
                  ///< See \cite lee-seung-algs.
    };

    /**
     * Matrix used in the computations.
     *
     * @TODO: Templatize the matrix elements (int, double, float, etc.)
     */
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    /**
     * @brief Compute nonnegative matrix factorization of a matrix using
     * Euclidean distance.
     *
     * According to the NMF model, nonnegative matrix X is factorized into two
     * nonnegative matrices W and H such that
     *
     *          X = WH
     *
     * where X is (m x n), W is (m x r) and H is (r x n) nonnegative matrices.
     *
     * @param X (m x n) nonnegative matrix.
     * @param r Inner dimension of the factorization. Must be positive.
     * @param variant NMF variant. See enum bnmf_algs::NMFVariant.
     * @param max_iter Maximum number of iterations. If set to 0, the algorithm
     *                  runs until convergence. Must be nonnegative.
     * @param epsilon Convergence measure. For steps
     *                 \f$ i \f$ and \f$ i+1 \f$, if
     *                 \f$ |X-WH|_{i+1} - |X-WH|_i \leq epsilon \f$, the
     *                 algorithm is assumed to have converged. Must be
     *                 nonnegative.
     *
     * @remark The algorithm terminates when at least one of the termination
     *          conditions related to max_iter and epsilon is satisfied.
     * @return std::pair of W and H matrices.
     *
     * @author Esref Ozdemir
     */
    std::pair<Matrix, Matrix> nmf(const Matrix& X,
                                  long r,
                                  NMFVariant variant,
                                  int max_iter=250,
                                  double epsilon=std::numeric_limits<double>::epsilon());

    /**
     * Compute the Euclidean cost as defined in \cite lee-seung-algs.
     *
     * Euclidean cost is defined as
     *
     * \f$ \sum_{ij} (A_{ij} - B_{ij})^2 \f$
     *
     * for matrices A and B of the same shape.
     *
     * @param A First matrix.
     * @param B Second matrix.
     *
     * @return Euclidean cost between A and B.
     *
     * @remark If matrices A and B have different shape, the result is undefined.
     * @author Esref Ozdemir
     */
    double euclidean_cost(const bnmf_algs::Matrix& A, const bnmf_algs::Matrix& B);

    /**
     * Compute the KL-divergence cost as defined in \cite lee-seung-algs.
     *
     * This function computes the KL-divergence cost from A to B which is defined
     * as
     *
     * \f$ \sum_{ij} \big( A_{ij}log(\frac{A_{ij}}{B_{ij}}) - A_{ij} + B_{ij} \big)\f$
     *
     * for matrices A and B of the same shape. If \f$ A_{ij} = 0 \f$ or
     * \f$ B_{ij} = 0\f$, then the first term in the equation is not calculated
     * for that element pair.
     *
     * @param A First matrix.
     * @param B Second matrix.
     *
     * @return KL-divergence cost from A to B.
     *
     * @remark If matrices A and B have different shape, the result is undefined.
     * @author Esref Ozdemir
     */
    double kl_cost(const bnmf_algs::Matrix& A, const bnmf_algs::Matrix& B);
}

