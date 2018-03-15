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
static std::string nmf_check_parameters(const matrix_t& X, size_t r) {
    if ((X.array() < 0).any()) {
        return "X matrix has negative entries";
    }
    if (r == 0) {
        return "r must be positive";
    }
    return "";
}

double nmf::beta_divergence(double x, double y, double beta) {
    constexpr double eps = std::numeric_limits<double>::epsilon();

    if (std::abs(beta) <= eps) {
        return x / y - std::log(x / y) - 1;
    } else if (std::abs(beta - 1) <= eps) {
        return x * std::log(x / y) - x + y;
    }
    double nom = std::pow(x, beta) + (beta - 1) * std::pow(y, beta) -
                 beta * x * std::pow(y, beta - 1);
    return nom / (beta * (beta - 1));
}

/**
 * @brief Compute delta1 function as defined in Derivation Sketch for NMF Using
 * \f$\beta\f$-divergence.
 *
 * delta1 function is defined as
 *
 * \f$ \sum_j Y(i, j)H(k, j) \f$.
 *
 * @param H H matrix in NMF where \f$X \sim WH\f$.
 * @param Y Matrix to apply delta1 function in-place.
 */
static void delta1(const matrix_t& H, matrix_t& Y) {

}

/**
 * @brief Compute delta2 function as defined in Derivation Sketch for NMF Using
 * \f$\beta\f$-divergence.
 *
 * delta2 function is defined as
 *
 * \f$ \sum_i Y(i, j)W(i, k)\f$.
 *
 * @param W W matrix in NMF where \f$X \sim WH\f$.
 * @param Y Matrix to apply delta2 function in-place.
 */
static void delta2(const matrix_t& W, matrix_t& Y) {}

std::pair<matrix_t, matrix_t> nmf::nmf(const matrix_t& X, size_t r, double beta,
                                       size_t max_iter) {
    const auto m = static_cast<size_t>(X.rows());
    const auto n = static_cast<size_t>(X.cols());

    {
        auto error_msg = nmf_check_parameters(X, r);
        if (!error_msg.empty()) {
            throw std::invalid_argument(error_msg);
        }
    }

    if (X.isZero(0)) {
        return std::make_pair(matrix_t::Zero(m, r), matrix_t::Zero(r, n));
    }

    matrix_t W = matrix_t::Random(m, r) + matrix_t::Ones(m, r);
    matrix_t H = matrix_t::Random(r, n) + matrix_t::Ones(r, n);

    if (max_iter == 0) {
        return std::make_pair(W, H);
    }

    double p = 2 - beta;
    while (max_iter-- > 0) {
        matrix_t X_hat = W * H;

        matrix_t nom = X.array() / X_hat.array().pow(p);
        matrix_t denom = X_hat.array().pow(1 - p);
        delta1(H, nom);
        delta1(H, denom);

        W = W.array() * nom.array() / denom.array();

        X_hat = W * H;

        nom = X.array() / X_hat.array().pow(p);
        denom = X_hat.array().pow(1 - p);
        delta2(W, nom);
        delta2(W, denom);

        H = H.array() * nom.array() / denom.array();
    }
    return std::make_pair(W, H);
}
