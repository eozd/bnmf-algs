#include "nmf/nmf.hpp"
#include <chrono>
#include <iostream>

#include <cmath>
#include <limits>
#include <stdexcept>
using namespace std::chrono;

using namespace bnmf_algs;

/**
 * @brief Check that given parameters satisfy the constraints as specified in
 * bnmf_algs::nmf.
 *
 * @return Error message. If there isn't any error, then returns "".
 */
static std::string nmf_check_parameters(const matrix_t& X, size_t r) {
    if ((X.array() < 0).any()) {
        return "X matrix has negative entries";
    }
    if (r == 0) {
        return "r must be positive";
    }
    return "";
}

double nmf::beta_divergence(double x, double y, double beta, double eps) {
    if (std::abs(beta) <= std::numeric_limits<double>::epsilon()) {
        // Itakuro Saito
        if (std::abs(x) <= std::numeric_limits<double>::epsilon()) {
            return -1;
        }
        return x / (y + eps) - std::log(x / (y + eps)) - 1;
    } else if (std::abs(beta - 1) <= std::numeric_limits<double>::epsilon()) {
        // KL
        double logpart;
        if (std::abs(x) <= std::numeric_limits<double>::epsilon()) {
            logpart = 0;
        } else {
            logpart = x * std::log(x / (y + eps));
        }
        return logpart - x + y;
    }

    // general case
    double nom = std::pow(x, beta) + (beta - 1) * std::pow(y, beta) -
                 beta * x * std::pow(y, beta - 1);
    // beta != 0 && beta != 1
    return nom / (beta * (beta - 1));
}

std::pair<matrix_t, matrix_t> nmf::nmf(const matrix_t& X, size_t r, double beta,
                                       size_t max_iter) {
    const auto x = static_cast<size_t>(X.rows());
    const auto y = static_cast<size_t>(X.cols());
    const auto z = r;

    {
        auto error_msg = nmf_check_parameters(X, r);
        if (!error_msg.empty()) {
            throw std::invalid_argument(error_msg);
        }
    }

    if (X.isZero(0)) {
        return std::make_pair(matrix_t::Zero(x, z), matrix_t::Zero(z, y));
    }

    matrix_t W = matrix_t::Random(x, z) + matrix_t::Ones(x, z);
    matrix_t H = matrix_t::Random(z, y) + matrix_t::Ones(z, y);

    if (max_iter == 0) {
        return std::make_pair(W, H);
    }

    const double p = 2 - beta;
    const double eps = std::numeric_limits<double>::epsilon();
    const auto p_int = (int)p;
    const bool is_p_int = std::abs(p_int - p) <= eps;

    matrix_t X_hat, nom_W(x, z), denom_W(x, z), nom_H(z, y), denom_H(z, y);
    while (max_iter-- > 0) {
        // update approximation
        X_hat = W * H;

        // update W
        if (is_p_int && p == 0) {
            // Euclidean NMF
            nom_W = X * H.transpose();
            denom_W = X_hat * H.transpose();
        } else if (is_p_int && p == 1) {
            // KL NMF
            nom_W = (X.array() / X_hat.array()).matrix() * H.transpose();
            denom_W = matrix_t::Ones(x, y) * H.transpose();
        } else if (is_p_int && p == 2) {
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
        if (is_p_int && p == 0) {
            // Euclidean NMF
            nom_H = W.transpose() * X;
            denom_H = W.transpose() * X_hat;
        } else if (is_p_int && p == 1) {
            // KL NMF
            nom_H = W.transpose() * (X.array() / X_hat.array()).matrix();
            denom_H = W.transpose() * matrix_t::Ones(x, y);
        } else if (is_p_int && p == 2) {
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
