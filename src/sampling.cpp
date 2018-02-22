#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>
#include <iostream>

#include "sampling.hpp"
#include "wrappers.hpp"

/**
 * @brief Check if any of the dimensions is non-positive.
 *
 * @throws std::invalid_argument if any of the dimensions is non-positive.
 *
 * @param x First dimension.
 * @param y Second dimension.
 * @param z Third dimension.
 *
 * @author Esref Ozdemir
 */
static void ensure_positive_dimensions(long x, long y, long z) {
    if (x <= 0 || y <= 0 || z <= 0) {
        throw std::invalid_argument(
            "Non-positive shape: (" + std::to_string(x) + ", " +
            std::to_string(y) + ", " + std::to_string(z) + ")");
    }
}

/**
 * @brief Check if shapes of \f$W\f$, \f$H\f$ and \f$L\f$ is incompatible.
 *
 * According to the BNMF model, \f$W\f$ is an \f$x \times y\f$ matrix,
 * \f$H\f$ is an \f$z \times y\f$ matrix and \f$L\f$ is a vector of size
 * \f$y\f$.
 *
 * @throws std::invalid_argument if the dimensions are not compatible with BNMF
 * model.
 *
 * @param prior_W Prior matrix \f$W\f$.
 * @param prior_H Prior matrix \f$H\f$.
 * @param prior_L Prior vector \f$L\f$.
 *
 * @author Esref Ozdemir
 */
static void ensure_compatible_dimensions(const bnmf_algs::matrix_t& prior_W,
                                         const bnmf_algs::matrix_t& prior_H,
                                         const bnmf_algs::vector_t& prior_L) {
    if (prior_W.cols() != prior_H.rows()) {
        throw std::invalid_argument("Incompatible dimensions: W=(" +
                                    std::to_string(prior_W.rows()) + ", " +
                                    std::to_string(prior_W.cols()) + ") H=(" +
                                    std::to_string(prior_H.rows()) + ", " +
                                    std::to_string(prior_H.cols()) + ")");
    }
    if (prior_H.cols() != prior_L.cols()) {
        throw std::invalid_argument("Incompatible dimensions: H=(" +
                                    std::to_string(prior_H.rows()) + ", " +
                                    std::to_string(prior_H.cols()) + ") L=(" +
                                    std::to_string(prior_L.cols()) + ")");
    }
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
 * @throws std::invalid_argument If the number of Dirichlet parameters is not
 * equal to corresponding dimensions.
 *
 * @author Esref Ozdemir
 */
static void
ensure_compatible_dirichlet_parameters(long x, long z,
                                       const std::vector<double>& alpha,
                                       const std::vector<double>& beta) {
    if (alpha.size() != x) {
        throw std::invalid_argument("Number of dirichlet parameters alpha (" +
                                    std::to_string(alpha.size()) +
                                    ") not equal to x (" + std::to_string(x) +
                                    ")");
    }
    if (beta.size() != z) {
        throw std::invalid_argument("Number of dirichlet parameters beta (" +
                                    std::to_string(beta.size()) +
                                    ") not equal to z (" + std::to_string(z) +
                                    ")");
    }
}

std::tuple<bnmf_algs::matrix_t, bnmf_algs::matrix_t, bnmf_algs::vector_t>
bnmf_algs::bnmf_priors(const shape<3>& tensor_shape, double a, double b,
                       const std::vector<double>& alpha,
                       const std::vector<double>& beta) {
    long x = tensor_shape[0], y = tensor_shape[1], z = tensor_shape[2];
    // TODO: Preprocesor options to disable checks
    ensure_positive_dimensions(x, y, z);
    ensure_compatible_dirichlet_parameters(x, z, alpha, beta);

    auto rand_gen = make_gsl_rng(gsl_rng_taus);
    // generate prior_L
    vector_t prior_L(y);
    for (int i = 0; i < y; ++i) {
        prior_L(i) = gsl_ran_gamma(rand_gen.get(), a, b);
    }
    // generate prior_W
    matrix_t prior_W(x, z);
    vector_t dirichlet_variates(x);
    for (int i = 0; i < z; ++i) {
        gsl_ran_dirichlet(rand_gen.get(), x, alpha.data(),
                          dirichlet_variates.data());

        for (int j = 0; j < x; ++j) {
            prior_W(j, i) = dirichlet_variates(j);
        }
    }
    // generate prior_H
    matrix_t prior_H(z, y);
    dirichlet_variates = vector_t(z);
    for (int i = 0; i < y; ++i) {
        gsl_ran_dirichlet(rand_gen.get(), z, beta.data(),
                          dirichlet_variates.data());

        for (int j = 0; j < z; ++j) {
            prior_H(j, i) = dirichlet_variates(j);
        }
    }

    return std::tuple<matrix_t, matrix_t, vector_t>(prior_W, prior_H, prior_L);
}

bnmf_algs::tensor3d_t bnmf_algs::sample_S(const matrix_t& prior_W,
                                          const matrix_t& prior_H,
                                          const vector_t& prior_L) {
    long x = prior_W.rows();
    long y = prior_L.cols();
    long z = prior_H.rows();
    // TODO: Preprocesor options to disable checks
    ensure_positive_dimensions(x, y, z);
    ensure_compatible_dimensions(prior_W, prior_H, prior_L);

    // TODO: Is it OK to construct this rng object every time?
    auto rand_gen = make_gsl_rng(gsl_rng_taus);
    tensor3d_t sample(x, y, z);
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
static double compute_first_term(const bnmf_algs::tensor3d_t& S,
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
            sum += alpha[i] * y;
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
static double compute_second_term(const bnmf_algs::tensor3d_t& S,
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
            sum += beta[k] * x;
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
static double compute_third_term(const bnmf_algs::tensor3d_t& S, double a,
                                 double b) {
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
static double compute_fourth_term(const bnmf_algs::tensor3d_t& S) {
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

double bnmf_algs::log_marginal_S(const tensor3d_t& S, double a, double b,
                                 const std::vector<double>& alpha,
                                 const std::vector<double>& beta) {
    return compute_first_term(S, alpha) + compute_second_term(S, beta) +
           compute_third_term(S, a, b) - compute_fourth_term(S);
}
