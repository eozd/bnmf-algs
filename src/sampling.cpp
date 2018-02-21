#include <gsl/gsl_randist.h>
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
 */
void ensure_positive_dimensions(long x, long y, long z) {
    if (x <= 0 || y <= 0 || z <= 0) {
        throw std::invalid_argument("Non-positive shape: (" +
                                    std::to_string(x) + ", " +
                                    std::to_string(y) + ", " +
                                    std::to_string(z) + ")");
    }
}

/**
 * @brief Check if shapes of \f$W\f$, \f$H\f$ and \f$L\f$ is incompatible.
 *
 * According to the BNMF model, \f$W\f$ is an \f$x \times y\f$ matrix,
 * \f$H\f$ is an \f$z \times y\f$ matrix and \f$L\f$ is a vector of size \f$y\f$.
 *
 * @throws std::invalid_argument if the dimensions are not compatible with BNMF
 * model.
 *
 * @param prior_W Prior matrix \f$W\f$.
 * @param prior_H Prior matrix \f$H\f$.
 * @param prior_L Prior vector \f$L\f$.
 */
void ensure_compatible_dimensions(const bnmf_algs::matrix_t& prior_W,
                                  const bnmf_algs::matrix_t& prior_H,
                                  const bnmf_algs::vector_t& prior_L) {
    if (prior_W.cols() != prior_H.rows()) {
        throw std::invalid_argument("Incompatible dimensions: W=(" +
                                            std::to_string(prior_W.rows()) +
                                            ", " + std::to_string(prior_W.cols()) +
                                            ") H=(" +
                                            std::to_string(prior_H.rows()) +
                                            ", " + std::to_string(prior_H.cols()) +
                                            ")");
    }
    if (prior_H.cols() != prior_L.cols()) {
        throw std::invalid_argument("Incompatible dimensions: H=(" +
                                    std::to_string(prior_H.rows()) +
                                    ", " + std::to_string(prior_H.cols()) +
                                    ") L=(" + std::to_string(prior_L.cols()) +
                                    ")");
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
 */
void ensure_compatible_dirichlet_parameters(long x,
                                            long z,
                                            const std::vector<double>& alpha,
                                            const std::vector<double>& beta) {
    if (alpha.size() != x) {
        throw std::invalid_argument("Number of dirichlet parameters alpha (" +
                                            std::to_string(alpha.size()) +
                                            ") not equal to x (" +
                                            std::to_string(x) + ")");
    }
    if (beta.size() != z) {
        throw std::invalid_argument("Number of dirichlet parameters beta (" +
                                    std::to_string(beta.size()) +
                                    ") not equal to z (" +
                                    std::to_string(z) + ")");

    }
}

std::tuple<bnmf_algs::matrix_t, bnmf_algs::matrix_t, bnmf_algs::vector_t>
bnmf_algs::bnmf_priors(const shape<3>& tensor_shape,
                       double a,
                       double b,
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
        gsl_ran_dirichlet(rand_gen.get(), x, alpha.data(), dirichlet_variates.data());

        for (int j = 0; j < x; ++j) {
            prior_W(j, i) = dirichlet_variates(j);
        }
    }
    // generate prior_H
    matrix_t prior_H(z, y);
    dirichlet_variates = vector_t(z);
    for (int i = 0; i < y; ++i) {
        gsl_ran_dirichlet(rand_gen.get(), z, beta.data(), dirichlet_variates.data());

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
                mu = prior_W(i, k)*prior_H(k, j)*prior_L(j);
                sample(i, j, k) = gsl_ran_poisson(rand_gen.get(), mu);
            }
        }
    }
    return sample;
}
