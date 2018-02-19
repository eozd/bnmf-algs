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
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> nmf(const Eigen::MatrixXd& X,
                                                    long r,
                                                    NMFVariant variant,
                                                    int max_iter=1000,
                                                    double epsilon=std::numeric_limits<double>::epsilon());
}

