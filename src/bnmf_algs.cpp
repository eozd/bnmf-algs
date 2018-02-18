#include "bnmf_algs.hpp"

#include <stdexcept>
#include <limits>

/**
 * Check that given parameters satisfy the constraints as specified in @ref bnmf_algs::nmf_euclidean.
 */
void check_parameters(const Eigen::MatrixXd& X, long r, int max_iter, double epsilon) {
    if ((X.array() < 0).any()) {
        throw std::invalid_argument("X matrix has negative entries");
    }
    if (r <= 0) {
        throw std::invalid_argument("r must be positive");
    }
    if (max_iter < 0) {
        throw std::invalid_argument("max_iter must be nonnegative");
    }
    if (epsilon < 0) {
        throw std::invalid_argument("epsilon must be nonnegitave");
    }
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> bnmf_algs::nmf_euclidean(const Eigen::MatrixXd& X, long r, int max_iter, double epsilon) {
    using namespace Eigen;
    const long m = X.rows();
    const long n = X.cols();

    check_parameters(X, r, max_iter, epsilon);

    // special case (X == 0)
    if (X.maxCoeff() == 0.0) {
        return {MatrixXd::Zero(m, r), MatrixXd::Zero(r, n)};
    }

    // initialize
    MatrixXd W = MatrixXd::Random(m, r) + MatrixXd::Ones(m, r);
    MatrixXd H = MatrixXd::Random(r, n) + MatrixXd::Ones(r, n);

    double cost = 0., prev_cost;
    bool no_stop = max_iter == 0;
    while (no_stop || max_iter-- > 0) {
        MatrixXd curr_approx = W*H;

        // update costs
        prev_cost = cost;
        cost = (X - curr_approx).norm();

        // check cost convergence
        if (std::abs(cost - prev_cost) < epsilon) {
            break;
        }
        // H update
        MatrixXd numer = W.transpose()*X;
        MatrixXd denom = W.transpose()*curr_approx;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < r; ++i) {
                H(i, j) *= numer(i, j)/denom(i, j);
            }
        }

        // W update
        numer = X*H.transpose();
        denom = W*H*H.transpose();
        for (int j = 0; j < r; ++j) {
            for (int i = 0; i < m; ++i) {
                W(i, j) *= numer(i, j)/denom(i, j);
            }
        }
    }
    return {W, H};
}
