#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>
#include <iostream>

#include "allocation_model/sampling.hpp"
#include "util/wrappers.hpp"

using namespace bnmf_algs;

/**
 * @brief Check if shapes of \f$W\f$, \f$H\f$ and \f$L\f$ is incompatible.
 *
 * According to the BNMF model, \f$W\f$ is an \f$x \times y\f$ matrix,
 * \f$H\f$ is an \f$z \times y\f$ matrix and \f$L\f$ is a vector of size
 * \f$y\f$.
 *
 * @param prior_W Prior matrix \f$W\f$.
 * @param prior_H Prior matrix \f$H\f$.
 * @param prior_L Prior vector \f$L\f$.
 *
 * @return Error message. If there isn't any error, returns "".
 */
static std::string ensure_compatible_dimensions(const matrix_t& prior_W,
                                                const matrix_t& prior_H,
                                                const vector_t& prior_L) {
    if (prior_W.cols() != prior_H.rows()) {
        return "Incompatible dimensions: W=(" + std::to_string(prior_W.rows()) +
               ", " + std::to_string(prior_W.cols()) + ") H=(" +
               std::to_string(prior_H.rows()) + ", " +
               std::to_string(prior_H.cols()) + ")";
    }
    if (prior_H.cols() != prior_L.cols()) {
        return "Incompatible dimensions: H=(" + std::to_string(prior_H.rows()) +
               ", " + std::to_string(prior_H.cols()) + ") L=(" +
               std::to_string(prior_L.cols()) + ")";
    }
    return "";
}

/**
 * @brief Check if the number of given Dirichlet parameters equal to
 * corresponding dimensions.
 *
 * @param x First dimension of the sample tensor.
 * @param z Third dimension of the sample tensor.
 * @param alpha Dirichlet parameters for \f$W\f$.
 * @param beta Dirichlet parameters for \f$H\f$.
 *
 * @return Error message. If there isn't any errors, returns "".
 */
static std::string
ensure_compatible_dirichlet_parameters(size_t x, size_t z,
                                       const std::vector<double>& alpha,
                                       const std::vector<double>& beta) {
    if (alpha.size() != x) {
        return "Number of dirichlet parameters alpha (" +
               std::to_string(alpha.size()) + ") not equal to x (" +
               std::to_string(x) + ")";
    }
    if (beta.size() != z) {
        return "Number of dirichlet parameters beta (" +
               std::to_string(beta.size()) + ") not equal to z (" +
               std::to_string(z) + ")";
    }
    return "";
}

std::tuple<matrix_t, matrix_t, vector_t> allocation_model::bnmf_priors(
    const shape<3>& tensor_shape,
    const allocation_model::AllocModelParams& model_params) {

    size_t x = tensor_shape[0], y = tensor_shape[1], z = tensor_shape[2];
    // TODO: Preprocesor options to disable checks
    {
        auto error_msg = ensure_compatible_dirichlet_parameters(
            x, z, model_params.alpha, model_params.beta);
        if (!error_msg.empty()) {
            throw std::invalid_argument(error_msg);
        }
    }

    auto rand_gen = util::make_gsl_rng(gsl_rng_taus);
    // generate prior_L
    vector_t prior_L(y);
    for (size_t i = 0; i < y; ++i) {
        prior_L(i) =
            gsl_ran_gamma(rand_gen.get(), model_params.a, model_params.b);
    }
    // generate prior_W
    matrix_t prior_W(x, z);
    vector_t dirichlet_variates(x);
    for (size_t i = 0; i < z; ++i) {
        gsl_ran_dirichlet(rand_gen.get(), x, model_params.alpha.data(),
                          dirichlet_variates.data());

        for (size_t j = 0; j < x; ++j) {
            prior_W(j, i) = dirichlet_variates(j);
        }
    }
    // generate prior_H
    matrix_t prior_H(z, y);
    dirichlet_variates = vector_t(z);
    for (size_t i = 0; i < y; ++i) {
        gsl_ran_dirichlet(rand_gen.get(), z, model_params.beta.data(),
                          dirichlet_variates.data());

        for (size_t j = 0; j < z; ++j) {
            prior_H(j, i) = dirichlet_variates(j);
        }
    }

    return std::make_tuple(prior_W, prior_H, prior_L);
}

tensord<3> allocation_model::sample_S(const matrix_t& prior_W,
                                      const matrix_t& prior_H,
                                      const vector_t& prior_L) {
    long x = prior_W.rows();
    long y = prior_L.cols();
    long z = prior_H.rows();
    // TODO: Preprocesor options to disable checks
    {
        auto error_msg =
            ensure_compatible_dimensions(prior_W, prior_H, prior_L);
        if (error_msg != "") {
            throw std::invalid_argument(error_msg);
        }
    }

    // TODO: Is it OK to construct this rng object every time?
    auto rand_gen = util::make_gsl_rng(gsl_rng_taus);
    tensord<3> sample(x, y, z);
    double mu;
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            for (int k = 0; k < z; ++k) {
                mu = prior_W(i, k) * prior_H(k, j) * prior_L(j);
                sample(i, j, k) = gsl_ran_poisson(rand_gen.get(), mu);
            }
        }
    }
    return sample;
}

/**
 * @brief Compute the first term of the sum when calculating log marginal of S.
 *
 * @param S Tensor S.
 * @param alpha Parameter vector of Dirichlet prior for matrix \f$W\f$ of size
 * \f$x\f$.
 *
 * @return Value of the first term
 */
static double compute_first_term(const tensord<3>& S,
                                 const std::vector<double>& alpha) {
    long x = S.dimension(0), y = S.dimension(1), z = S.dimension(2);

    double log_gamma_sum =
        gsl_sf_lngamma(std::accumulate(alpha.begin(), alpha.end(), 0.0));
    double sum_log_gamma = 0;
    std::for_each(alpha.begin(), alpha.end(), [&sum_log_gamma](double alpha_i) {
        sum_log_gamma += gsl_sf_lngamma(alpha_i);
    });

    double first = log_gamma_sum * z - sum_log_gamma * z;
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
    return first;
}

/**
 * @brief Compute the second term of the sum when calculating log marginal of S.
 *
 * @param S Tensor S.
 * @param beta Parameter vector of Dirichlet prior for matrix \f$H\f$ of size
 * \f$z\f$.
 *
 * @return Value of the second term
 */
static double compute_second_term(const tensord<3>& S,
                                  const std::vector<double>& beta) {
    long x = S.dimension(0), y = S.dimension(1), z = S.dimension(2);

    double log_gamma_sum =
        gsl_sf_lngamma(std::accumulate(beta.begin(), beta.end(), 0.0));
    double sum_log_gamma = 0;
    std::for_each(beta.begin(), beta.end(), [&sum_log_gamma](double beta_k) {
        sum_log_gamma += gsl_sf_lngamma(beta_k);
    });

    double second = log_gamma_sum * y - sum_log_gamma * y;
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
    return second;
}

/**
 * @brief Compute the third term of the sum when calculating log marginal of S.
 *
 * @param S Tensor S.
 * @param a Shape parameter of Gamma distribution.
 * @param b Rate parameter of Gamma distribution.
 *
 * @return Value of the third term
 */
static double compute_third_term(const tensord<3>& S, double a, double b) {
    long x = S.dimension(0), y = S.dimension(1), z = S.dimension(2);

    double log_gamma = -gsl_sf_lngamma(a);
    double a_log_b = a * std::log(b);

    double third = log_gamma * y + a_log_b * y;
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
    return third;
}

/**
 * @brief Compute the fourth term of the sum when calculating log marginal of S.
 *
 * @param S Tensor S.
 *
 * @return Value of the fourth term
 */
static double compute_fourth_term(const tensord<3>& S) {
    long x = S.dimension(0), y = S.dimension(1), z = S.dimension(2);

    double fourth = 0;
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            for (int k = 0; k < z; ++k) {
                fourth += gsl_sf_lngamma(S(i, j, k) + 1);
            }
        }
    }
    return fourth;
}

double allocation_model::log_marginal_S(const tensord<3>& S,
                                        const AllocModelParams& model_params) {
    if (model_params.alpha.size() != static_cast<size_t>(S.dimension(0))) {
        throw std::invalid_argument(
            "Number of alpha parameters must be equal to S.dimension(0)");
    }
    if (model_params.beta.size() != static_cast<size_t>(S.dimension(2))) {
        throw std::invalid_argument(
            "Number of beta parameters must be equal to S.dimension(2)");
    }

    return compute_first_term(S, model_params.alpha) +
           compute_second_term(S, model_params.beta) +
           compute_third_term(S, model_params.a, model_params.b) -
           compute_fourth_term(S);
}

using namespace allocation_model;

SampleOnesComputer::SampleOnesComputer(const matrix_t& X, bool replacement,
                                       size_t n)
    : X(X), replacement(replacement), n(n),
      X_cumsum(vector_t(X.cols() * X.rows())),
      X_cols(static_cast<size_t>(X.cols())), X_sum(X.array().sum()),
      rnd_gen(util::make_gsl_rng(gsl_rng_taus)) {

    auto X_arr = X.array();
    X_cumsum(0) = X_arr(0);
    for (int i = 1; i < X_arr.size(); ++i) {
        X_cumsum(i) = X_cumsum(i - 1) + X_arr(i);
    }

    if (!replacement) {
        X_sum = std::floor(X_sum);
    }
}

void SampleOnesComputer::operator()(size_t curr_step,
                                    std::pair<int, int>& prev_val) {
    double u = gsl_ran_flat(rnd_gen.get(), 0, 1);

    vector_t cum_prob;
    if (replacement) {
        cum_prob = X_cumsum / X_sum;
    } else {
        cum_prob = X_cumsum / (X_sum - (curr_step - 1));
    }

    double* begin = cum_prob.data();
    double* end = cum_prob.data() + cum_prob.cols();
    auto it = std::upper_bound(begin, end, u);

    long m = it - begin;

    if (!replacement) {
        vector_t diff = vector_t::Constant(X_cumsum.cols(), 1);
        diff.head(m) *= 0;
        X_cumsum -= diff;
    }

    prev_val.first = static_cast<int>(m / X_cols);
    prev_val.second = static_cast<int>(m % X_cols);
}

util::Generator<std::pair<int, int>, SampleOnesComputer>
allocation_model::sample_ones(const matrix_t& X, bool replacement, size_t n) {
    size_t num_samples;
    if (replacement) {
        num_samples = n;
    } else {
        num_samples = static_cast<size_t>(X.array().sum());
    }
    // don't know the initial value
    std::pair<int, int> init_val;
    util::Generator<std::pair<int, int>, SampleOnesComputer> gen(
        init_val, num_samples + 1, SampleOnesComputer(X, replacement, n));

    // get rid of the first empty value
    ++gen.begin();

    return gen;
}
