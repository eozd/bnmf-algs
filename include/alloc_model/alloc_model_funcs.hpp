#pragma once

#include "alloc_model/alloc_model_params.hpp"
#include "defs.hpp"
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>
#include <tuple>
#include "util/generator.hpp"
#include <vector>

namespace bnmf_algs {

/**
 * @brief Namespace that contains functions related to Allocation Model \cite
 * kurutmazbayesian.
 */
namespace alloc_model {

/**
 * @brief Return prior matrices W, H and vector L according to the Bayesian
 * NMF allocation model using distribution parameters.
 *
 * According to the Bayesian NMF allocation model \cite kurutmazbayesian,
 *
 * \f$L_j \sim \mathcal{G}(a, b) \qquad W_{:k} \sim \mathcal{D}(\alpha)
 * \qquad H_{:j} \sim \mathcal{D}(\beta)\f$.
 *
 * @tparam Value type of the matrix/vector objects.
 * @param tensor_shape Shape of the sample tensor \f$x \times y \times z\f$.
 * @param model_params Allocation Model parameters. See
 * bnmf_algs::alloc_model::Params<double>.
 *
 * @return std::tuple of \f$<W_{x \times z}, H_{z \times y}, L_y>\f$.
 *
 * @throws assertion error if number of alpha parameters is not \f$x\f$,
 * if number of beta parameters is not \f$z\f$.
 */
template <typename T>
std::tuple<matrix_t<T>, matrix_t<T>, vector_t<T>>
bnmf_priors(const shape<3>& tensor_shape, const Params<double>& model_params) {
    size_t x = tensor_shape[0], y = tensor_shape[1], z = tensor_shape[2];

    BNMF_ASSERT(model_params.alpha.size() == x,
                "Number of dirichlet parameters alpha must be equal to x");
    BNMF_ASSERT(model_params.beta.size() == z,
                "Number of dirichlet parameters beta must be equal to z");

    util::gsl_rng_wrapper rand_gen(gsl_rng_alloc(gsl_rng_taus), gsl_rng_free);

    // generate prior_L
    vector_t<T> prior_L(y);
    for (size_t i = 0; i < y; ++i) {
        prior_L(i) =
            gsl_ran_gamma(rand_gen.get(), model_params.a, model_params.b);
    }

    // generate prior_W
    matrix_t<T> prior_W(x, z);
    vector_t<T> dirichlet_variates(x);
    for (size_t i = 0; i < z; ++i) {
        gsl_ran_dirichlet(rand_gen.get(), x, model_params.alpha.data(),
                          dirichlet_variates.data());

        for (size_t j = 0; j < x; ++j) {
            prior_W(j, i) = dirichlet_variates(j);
        }
    }

    // generate prior_H
    matrix_t<T> prior_H(z, y);
    dirichlet_variates = vector_t<T>(z);
    for (size_t i = 0; i < y; ++i) {
        gsl_ran_dirichlet(rand_gen.get(), z, model_params.beta.data(),
                          dirichlet_variates.data());

        for (size_t j = 0; j < z; ++j) {
            prior_H(j, i) = dirichlet_variates(j);
        }
    }

    return std::make_tuple(prior_W, prior_H, prior_L);
}

/**
 * @brief Sample a tensor S from generative Bayesian NMF model using the
 * given priors.
 *
 * According to the generative Bayesian NMF model \cite kurutmazbayesian,
 *
 * \f$S_{ijk} \sim \mathcal{PO}(W_{ik}H_{kj}L_{j})\f$.
 *
 * @tparam T Value type of the matrix/vector/tensor objects.
 * @param prior_W Prior matrix for \f$W\f$ of shape \f$x \times z\f$.
 * @param prior_H Prior matrix for \f$H\f$ of shape \f$z \times y\f$.
 * @param prior_L Prior vector for \f$L\f$ of size \f$y\f$.
 *
 * @return A sample tensor of size \f$x \times y \times z\f$ from
 * generative Bayesian NMF model.
 *
 * @throws assertion error if number of columns of prior_W is not equal to
 * number of rows of prior_H, if number of columns of prior_H is not equal to
 * number of columns of prior_L
 */
template <typename T>
tensor_t<T, 3> sample_S(const matrix_t<T>& prior_W, const matrix_t<T>& prior_H,
                        const vector_t<T>& prior_L) {
    auto x = static_cast<size_t>(prior_W.rows());
    auto y = static_cast<size_t>(prior_L.cols());
    auto z = static_cast<size_t>(prior_H.rows());

    BNMF_ASSERT(prior_W.cols() == prior_H.rows(),
                "Number of columns of W is different than number of rows of H");
    BNMF_ASSERT(prior_H.cols() == prior_L.cols(),
                "Number of columns of H is different than size of L");

    // TODO: Is it OK to construct this rng object every time?
    util::gsl_rng_wrapper rand_gen(gsl_rng_alloc(gsl_rng_taus), gsl_rng_free);
    tensor_t<T, 3> sample(x, y, z);
    double mu;
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            for (size_t k = 0; k < z; ++k) {
                mu = prior_W(i, k) * prior_H(k, j) * prior_L(j);
                sample(i, j, k) = gsl_ran_poisson(rand_gen.get(), mu);
            }
        }
    }
    return sample;
}

namespace details {

/**
 * @brief Compute the first term of the sum when calculating log marginal of S.
 *
 * @tparam T Value type of the tensor S.
 * @param S Tensor S.
 * @param alpha Parameter vector of Dirichlet prior for matrix \f$W\f$ of size
 * \f$x\f$.
 *
 * @return Value of the first term
 */
template <typename T>
double compute_first_term(const tensor_t<T, 3>& S,
                          const std::vector<double>& alpha) {
    const long x = S.dimension(0);
    const long y = S.dimension(1);
    const long z = S.dimension(2);

    // loggamma(sum(alpha))
    const double log_gamma_sum =
        gsl_sf_lngamma(std::accumulate(alpha.begin(), alpha.end(), 0.0));

    // sum(loggamma(alpha))
    std::vector<double> log_gamma_alpha(alpha.size());
    std::transform(
        alpha.begin(), alpha.end(), log_gamma_alpha.begin(),
        [](const double alpha_i) { return gsl_sf_lngamma(alpha_i); });

    const double sum_log_gamma =
        std::accumulate(log_gamma_alpha.begin(), log_gamma_alpha.end(), 0.0);

    double first = 0;
    #pragma omp parallel for reduction(+:first)
    for (int k = 0; k < z; ++k) {
        double sum = 0;
        for (int i = 0; i < x; ++i) {
            sum += alpha[i];
            for (int j = 0; j < y; ++j) {
                sum += S(i, j, k);
            }
        }
        first -= gsl_sf_lngamma(sum);

        for (int i = 0; i < x; ++i) {
            sum = alpha[i];
            for (int j = 0; j < y; ++j) {
                sum += S(i, j, k);
            }
            first += gsl_sf_lngamma(sum);
        }
    }

    double base = log_gamma_sum * z - sum_log_gamma * z;

    return base + first;
}

/**
 * @brief Compute the second term of the sum when calculating log marginal of S.
 *
 * @tparam T Value type of tensor S.
 * @param S Tensor S.
 * @param beta Parameter vector of Dirichlet prior for matrix \f$H\f$ of size
 * \f$z\f$.
 *
 * @return Value of the second term
 */
template <typename T>
double compute_second_term(const tensor_t<T, 3>& S,
                           const std::vector<double>& beta) {
    const long x = S.dimension(0);
    const long y = S.dimension(1);
    const long z = S.dimension(2);

    // loggamma(sum(beta))
    const double log_gamma_sum =
        gsl_sf_lngamma(std::accumulate(beta.begin(), beta.end(), 0.0));

    // sum(loggamma(beta))
    std::vector<double> log_gamma_beta(beta.size());
    std::transform(beta.begin(), beta.end(), log_gamma_beta.begin(),
                   [](const double beta) { return gsl_sf_lngamma(beta); });

    const double sum_log_gamma =
        std::accumulate(log_gamma_beta.begin(), log_gamma_beta.end(), 0.0);

    double second = 0;
    #pragma omp parallel for reduction(+:second)
    for (int j = 0; j < y; ++j) {
        double sum = 0;
        for (int k = 0; k < z; ++k) {
            sum += beta[k];
            for (int i = 0; i < x; ++i) {
                sum += S(i, j, k);
            }
        }
        second -= gsl_sf_lngamma(sum);

        for (int k = 0; k < z; ++k) {
            sum = beta[k];
            for (int i = 0; i < x; ++i) {
                sum += S(i, j, k);
            }
            second += gsl_sf_lngamma(sum);
        }
    }

    double base = log_gamma_sum * y - sum_log_gamma * y;
    return base + second;
}

/**
 * @brief Compute the third term of the sum when calculating log marginal of S.
 *
 * @tparam T Valuet type of tensor S.
 * @param S Tensor S.
 * @param a Shape parameter of Gamma distribution.
 * @param b Rate parameter of Gamma distribution.
 *
 * @return Value of the third term
 */
template <typename T>
double compute_third_term(const tensor_t<T, 3>& S, double a, double b) {
    const long x = S.dimension(0);
    const long y = S.dimension(1);
    const long z = S.dimension(2);

    const double log_gamma = -gsl_sf_lngamma(a);
    const double a_log_b = a * std::log(b);

    double third = 0;
    #pragma omp parallel for reduction(+:third)
    for (int j = 0; j < y; ++j) {
        double sum = a;
        for (int i = 0; i < x; ++i) {
            for (int k = 0; k < z; ++k) {
                sum += S(i, j, k);
            }
        }
        third += gsl_sf_lngamma(sum);
        third -= sum * std::log(b + 1);
    }

    double base = log_gamma * y + a_log_b * y;
    return base + third;
}

/**
 * @brief Compute the fourth term of the sum when calculating log marginal of S.
 *
 * @tparam T Value type of tensor S.
 * @param S Tensor S.
 *
 * @return Value of the fourth term
 */
template <typename T> double compute_fourth_term(const tensor_t<T, 3>& S) {
    const long x = S.dimension(0);
    const long y = S.dimension(1);
    const long z = S.dimension(2);

    double fourth = 0;
    #pragma omp parallel for reduction(+:fourth)
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            for (int k = 0; k < z; ++k) {
                fourth += gsl_sf_lngamma(S(i, j, k) + 1);
            }
        }
    }
    return fourth;
}
} // namespace details

/**
 * @brief Compute the log marginal of tensor S with respect to the given
 * distribution parameters.
 *
 * According to the Bayesian NMF allocation model \cite kurutmazbayesian,
 *
 * \f$\log{p(S)} = \log{p(W, H, L, S)} - \log{p(W, H, L | S)}\f$.
 *
 * @tparam Value type of tensor S.
 * @param S Tensor \f$S\f$ of size \f$x \times y \times z \f$.
 * @param model_params Allocation model parameters. See
 * bnmf_algs::alloc_model::Params<double>.
 *
 * @return log marginal \f$\log{p(S)}\f$.
 *
 * @throws assertion error if number of alpha parameters is not equal to
 * S.dimension(0), if number of beta parameters is not equal to S.dimension(2).
 */
template <typename T>
double log_marginal_S(const tensor_t<T, 3>& S,
                      const Params<double>& model_params) {
    BNMF_ASSERT(model_params.alpha.size() ==
                    static_cast<size_t>(S.dimension(0)),
                "Number of alpha parameters must be equal to S.dimension(0)");
    BNMF_ASSERT(model_params.beta.size() == static_cast<size_t>(S.dimension(2)),
                "Number of alpha parameters must be equal to S.dimension(2)");

    return details::compute_first_term(S, model_params.alpha) +
           details::compute_second_term(S, model_params.beta) +
           details::compute_third_term(S, model_params.a, model_params.b) -
           details::compute_fourth_term(S);
}
} // namespace alloc_model
} // namespace bnmf_algs
