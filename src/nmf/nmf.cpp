#include "nmf/nmf.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

using namespace bnmf_algs;

/**
 * @brief Check that given parameters satisfy the constraints as specified in
 * bnmf_algs::nmf.
 *
 * @return Error message. If there isn't any error, then returns "".
 */
static std::string nmf_check_parameters(const matrix_t& X, size_t r,
                                        double epsilon) {
    if ((X.array() < 0).any()) {
        return "X matrix has negative entries";
    }
    if (r == 0) {
        return "r must be positive";
    }
    if (epsilon <= 0) {
        return "epsilon must be positive";
    }
    return "";
}

double nmf::euclidean_cost(const matrix_t& A, const matrix_t& B) {
    double cost = 0;
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            cost += std::pow(A(i, j) - B(i, j), 2);
        }
    }
    return cost;
}

double nmf::kl_cost(const matrix_t& X, const matrix_t& WH) {
    double cost = 0;
    double x_elem, wh_elem;
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j) {
            x_elem = X(i, j);
            wh_elem = WH(i, j);
            cost += wh_elem - x_elem;
            if (std::abs(x_elem) > std::numeric_limits<double>::epsilon() &&
                std::abs(wh_elem) > std::numeric_limits<double>::epsilon()) {
                cost += X(i, j) * std::log(X(i, j) / WH(i, j));
            }
        }
    }
    return cost;
}

std::pair<matrix_t, matrix_t> nmf::nmf(const matrix_t& X, size_t r,
                                       NMFVariant variant, size_t max_iter,
                                       double tolerance) {
    using namespace Eigen;
    const size_t m = static_cast<size_t>(X.rows());
    const size_t n = static_cast<size_t>(X.cols());

    {
        auto error_msg = nmf_check_parameters(X, r, tolerance);
        if (!error_msg.empty()) {
            throw std::invalid_argument(error_msg);
        }
    }

    // special case (X == 0)
    if (X.isZero(0)) {
        return {matrix_t::Zero(m, r), matrix_t::Zero(r, n)};
    }

    // initialize
    matrix_t W = matrix_t::Random(m, r) + matrix_t::Ones(m, r);
    matrix_t H = matrix_t::Random(r, n) + matrix_t::Ones(r, n);

    double cost = 0., prev_cost;
    bool until_convergence = (max_iter == 0);
    matrix_t numer, denom, curr_approx, frac;
    while (until_convergence || max_iter-- > 0) {
        curr_approx = W * H;

        // update cost
        prev_cost = cost;
        if (variant == NMFVariant::Euclidean) {
            cost = euclidean_cost(X, curr_approx);
        } else if (variant == NMFVariant::KL) {
            cost = kl_cost(X, curr_approx);
        }

        // check cost convergence
        if (std::abs(cost - prev_cost) <= tolerance) {
            break;
        }

        // H update
        if (variant == NMFVariant::Euclidean) {
            numer = (W.transpose() * X).eval();
            denom = (W.transpose() * curr_approx).eval();
            for (size_t a = 0; a < r; ++a) {
                for (size_t mu = 0; mu < n; ++mu) {
                    H(a, mu) *= numer(a, mu) / denom(a, mu);
                }
            }
        } else if (variant == NMFVariant::KL) {
            frac = X.cwiseQuotient(curr_approx);
            for (size_t a = 0; a < r; ++a) {
                double denom_sum = 0;
                for (size_t i = 0; i < m; ++i) {
                    denom_sum += W(i, a);
                }

                for (size_t mu = 0; mu < n; ++mu) {
                    double numer_sum = 0;
                    for (size_t i = 0; i < m; ++i) {
                        numer_sum += W(i, a) * frac(i, mu);
                    }
                    H(a, mu) *= numer_sum / denom_sum;
                }
            }
        }

        // W update
        if (variant == NMFVariant::Euclidean) {
            numer = (X * H.transpose()).eval();
            denom = (W * (H * H.transpose())).eval();
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < r; ++j) {
                    W(i, j) *= numer(i, j) / denom(i, j);
                }
            }
        } else if (variant == NMFVariant::KL) {
            frac = X.cwiseQuotient(W * H);
            for (size_t i = 0; i < m; ++i) {
                for (size_t a = 0; a < r; ++a) {

                    double numer_sum = 0, denom_sum = 0;
                    for (size_t mu = 0; mu < n; ++mu) {
                        numer_sum += H(a, mu) * frac(i, mu);
                        denom_sum += H(a, mu);
                    }
                    W(i, a) *= numer_sum / denom_sum;
                }
            }
        }
    }
    return std::make_pair(W, H);
}
