#pragma once

#include "allocation_model/alloc_model_params.hpp"
#include "defs.hpp"
#include <tuple>
#include <util/generator.hpp>
#include <util/wrappers.hpp>
#include <vector>

namespace bnmf_algs {
/**
 * @brief Namespace that contains types and functions for internal computations.
 *
 * Functions and classes in this namespace are not meant for direct usage by the
 * users of bnmf-algs library; hence, behaviours of some functions may not be
 * as expected. Additionally, these functions and types are not directly tested;
 * their behaviour is verified through correct behaviour of the API functions
 * and types bnmf-algs library exposes.
 */
namespace details {

/**
 * @brief Computer type that will compute the next sample when its operator() is
 * invoked.
 *
 * SampleOnesComputer will generate the next std::pair<int, int> value from the
 * previous value given to its operator(). This operation is performed in-place.
 */
class SampleOnesComputer {
  public:
    /**
     * @brief Construct a new SampleOnesComputer.
     *
     * @param X Matrix \f$X\f$ that will be used during sampling.
     * @param replacement Whether to sample with replacement.
     */
    explicit SampleOnesComputer(const matrix_t& X, bool replacement);

    /**
     * @brief Function call operator that will compute the next sample in-place.
     *
     * After this function exits, the object referenced by prev_val will be
     * modified to store the next sample.
     *
     * Note that SampleOnesComputer object satisfies the
     * bnmf_algs::util::Generator and bnmf_algs::util::ComputationIterator
     * Computer interface. Hence, a sequence of samples may be generated
     * efficiently by using this Computer with bnmf_algs::util::Generator or
     * bnmf_algs::util::ComputationIterator.
     *
     * @param curr_step Current step of the sampling computation.
     * @param prev_val Previously sampled value to modify in-place.
     */
    void operator()(size_t curr_step, std::pair<int, int>& prev_val);

  private:
    bool replacement;

    // computation variables
  private:
    vector_t X_cumsum;
    long X_cols;
    double X_sum;
    util::gsl_rng_wrapper rnd_gen;
};
} // namespace details

/**
 * @brief Namespace that contains functions related to Allocation Model \cite
 * kurutmazbayesian.
 */
namespace allocation_model {
/**
 * @brief Return prior matrices W, H and vector L according to the Bayesian
 * NMF allocation model using distribution parameters.
 *
 * According to the Bayesian NMF allocation model \cite kurutmazbayesian,
 *
 * \f$L_j \sim \mathcal{G}(a, b) \qquad W_{:k} \sim \mathcal{D}(\alpha)
 * \qquad H_{:j} \sim \mathcal{D}(\beta)\f$.
 *
 * @param tensor_shape Shape of the sample tensor \f$x \times y \times z\f$.
 * @param model_params Allocation Model parameters. See
 * bnmf_algs::allocation_model::AllocModelParams.
 *
 * @return std::tuple of \f$<W_{x \times z}, H_{z \times y}, L_y>\f$.
 *
 * @throws std::invalid_argument if number of alpha parameters is not \f$x\f$,
 * if number of beta parameters is not \f$z\f$.
 */
std::tuple<matrix_t, matrix_t, vector_t>
bnmf_priors(const shape<3>& tensor_shape, const AllocModelParams& model_params);

/**
 * @brief Sample a tensor S from generative Bayesian NMF model using the
 * given priors.
 *
 * According to the generative Bayesian NMF model \cite kurutmazbayesian,
 *
 * \f$S_{ijk} \sim \mathcal{PO}(W_{ik}H_{kj}L_{j})\f$.
 *
 * @param prior_W Prior matrix for \f$W\f$ of shape \f$x \times z\f$.
 * @param prior_H Prior matrix for \f$H\f$ of shape \f$z \times y\f$.
 * @param prior_L Prior vector for \f$L\f$ of size \f$y\f$.
 *
 * @return A sample tensor of size \f$x \times y \times z\f$ from
 * generative Bayesian NMF model.
 *
 * @throws std::invalid_argument if number of columns of prior_W is not equal to
 * number of rows of prior_H, if number of columns of prior_H is not equal to
 * number of columns of prior_L
 */
tensord<3> sample_S(const matrix_t& prior_W, const matrix_t& prior_H,
                    const vector_t& prior_L);

/**
 * @brief Compute the log marginal of tensor S with respect to the given
 * distribution parameters.
 *
 * According to the Bayesian NMF allocation model \cite kurutmazbayesian,
 *
 * \f$\log{p(S)} = \log{p(W, H, L, S)} - \log{p(W, H, L | S)}\f$.
 *
 * @param S Tensor \f$S\f$ of size \f$x \times y \times z \f$.
 * @param model_params Allocation model parameters. See
 * bnmf_algs::allocation_model::AllocModelParams.
 *
 * @return log marginal \f$\log{p(S)}\f$.
 *
 * @throws std::invalid_argument if number of alpha parameters is not equal to
 * S.dimension(0), if number of beta parameters is not equal to S.dimension(2).
 */
double log_marginal_S(const tensord<3>& S,
                      const AllocModelParams& model_params);

/**
 * @brief Return a bnmf_algs::util::Generator that will generate a sequence of
 * samples by using details::SampleOnesComputer as its Computer.
 *
 * This function returns a bnmf_algs::util::Generator object that will produce
 * a sequence of samples by generating the samples using SampleOnesComputer.
 * Since the return value is a generator, only a single sample is stored in the
 * memory at a given time.
 *
 * @param X Matrix \f$X\f$ from the Allocation Model \cite kurutmazbayesian.
 * @see bnmf_algs::allocation_model
 * @param replacement If true, sample with replacement.
 * @param n If replacement is true, then n is the number of samples that will be
 * generated. If replacement is false, then floor of sum of \f$X\f$ many samples
 * will be generated.
 *
 * @return A bnmf_algs::util::Generator type that can be used to generate a
 * sequence of samples.
 *
 * @throws std::invalid_argument if matrix \f$X\f$ has no element, if matrix
 * \f$X\f$ contains negative entries.
 */
util::Generator<std::pair<int, int>, details::SampleOnesComputer>
sample_ones(const matrix_t& X, bool replacement = false, size_t n = 1);
} // namespace allocation_model
} // namespace bnmf_algs
