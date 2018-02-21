#include <gsl/gsl_randist.h>

#include "sampling.hpp"
#include "wrappers.hpp"

bnmf_algs::tensor3d_t bnmf_algs::sample_S(const std::tuple<int, int, int>& tensor_shape,
                                          double a,
                                          double b,
                                          const std::vector<double>& alpha,
                                          const std::vector<double>& beta) {
    auto rand_gen = make_gsl_rng(gsl_rng_taus);
    int x, y, z;
    std::tie(x, y, z) = tensor_shape;
    // TODO: parameter validation

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
        prior_W.block(0, i, x, 1) = dirichlet_variates;
    }
    // generate prior_H
    matrix_t prior_H(z, y);
    dirichlet_variates.resize(z);
    for (int i = 0; i < y; ++i) {
        gsl_ran_dirichlet(rand_gen.get(), z, beta.data(), dirichlet_variates.data());
        prior_H.block(0, i, z, 1) = dirichlet_variates;
    }

    return sample_S(prior_L, prior_W, prior_H);
}

bnmf_algs::tensor3d_t bnmf_algs::sample_S(const vector_t& prior_L,
                                          const matrix_t& prior_W,
                                          const matrix_t& prior_H) {
    // TODO: Is it OK to construct this rng object every time?
    auto rand_gen = make_gsl_rng(gsl_rng_taus);
    long x = prior_W.rows();
    long y = prior_L.cols();
    long z = prior_W.cols();

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
