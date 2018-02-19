#include "bnmf_algs.hpp"

#include <stdexcept>
#include <limits>
#include <cmath>
#include <iostream>

/**
 * Check that given parameters satisfy the constraints as specified in bnmf_algs::nmf.
 */
void nmf_check_parameters(const bnmf_algs::Matrix& X, long r, int max_iter, double epsilon) {
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

double bnmf_algs::euclidean_cost(const bnmf_algs::Matrix& A, const bnmf_algs::Matrix& B) {
    double cost = 0;
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            cost += std::pow(A(i, j) - B(i, j) , 2);
        }
    }
    return cost;
}

double bnmf_algs::kl_cost(const bnmf_algs::Matrix& X, const bnmf_algs::Matrix& WH) {
    double cost = 0;
    double x_elem, wh_elem;
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j) {
            x_elem = X(i, j);
            wh_elem = WH(i, j);
            cost += wh_elem - x_elem;
            if (std::abs(x_elem) > std::numeric_limits<double>::epsilon() &&
                    std::abs(wh_elem) > std::numeric_limits<double>::epsilon()) {
                cost += X(i, j)*std::log(X(i, j)/WH(i, j));
            }
        }
    }
    return cost;
}

std::pair<bnmf_algs::Matrix, bnmf_algs::Matrix> bnmf_algs::nmf(const bnmf_algs::Matrix& X,
                                                               long r,
                                                               bnmf_algs::NMFVariant variant,
                                                               int max_iter,
                                                               double epsilon) {
    using namespace Eigen;
    const long m = X.rows();
    const long n = X.cols();

    nmf_check_parameters(X, r, max_iter, epsilon);

    // special case (X == 0)
    if (X.isZero(0)) {
        return {bnmf_algs::Matrix::Zero(m, r), bnmf_algs::Matrix::Zero(r, n)};
    }

    // initialize
    bnmf_algs::Matrix W = bnmf_algs::Matrix::Random(m, r) + bnmf_algs::Matrix::Ones(m, r);
    bnmf_algs::Matrix H = bnmf_algs::Matrix::Random(r, n) + bnmf_algs::Matrix::Ones(r, n);

    double cost = 0., prev_cost;
    bool until_convergence = (max_iter == 0);
    bnmf_algs::Matrix numer, denom, curr_approx;
    while (until_convergence || max_iter-- > 0) {
        curr_approx = W*H;

        // update cost
        prev_cost = cost;
        if (variant == NMFVariant::Euclidean) {
            cost = bnmf_algs::euclidean_cost(X, curr_approx);
        } else if (variant == NMFVariant::KL) {
            cost = bnmf_algs::kl_cost(X, curr_approx);
        }

        // check cost convergence
        if (std::abs(cost - prev_cost) <= epsilon) {
            break;
        }

        // H update
        if (variant == NMFVariant::Euclidean) {
            numer = (W.transpose()*X).eval();
            denom = (W.transpose()*curr_approx).eval();
            for (int i = 0; i < r; ++i) {
                for (int j = 0; j < n; ++j) {
                    H(i, j) *= numer(i, j)/denom(i, j);
                }
            }
        } else if (variant == NMFVariant::KL) {

        }

        // W update
        if (variant == NMFVariant::Euclidean) {
            numer = (X*H.transpose()).eval();
            denom = (W*(H*H.transpose())).eval();
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < r; ++j) {
                    W(i, j) *= numer(i, j)/denom(i, j);
                }
            }
        } else if (variant == NMFVariant::KL) {

        }
    }
    return {W, H};
}
