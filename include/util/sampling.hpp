#pragma once

#include "defs.hpp"
#include "util/generator.hpp"
#include "util/wrappers.hpp"

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
namespace util {

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

/**
 * @brief Choose a random sample according to the given cumulative probability
 * distribution.
 *
 * This function chooses a sample (index) from the given cumulative probability
 * distribution. Given distribution does not have to be normalized and it is
 * not normalized inside this function. A sample is drawn uniformly from the
 * range [0, cum_prob(N -1)] and then the corresponding index is found using
 * binary search.
 *
 * @param gsl_rng GSL random number generator object.
 * @param cum_prob Cumulative probability (weight) distribution.
 * @return Index of the random sample.
 *
 * @remark Complexity is \f$\Theta(logn)\f$.
 */
size_t choice(const bnmf_algs::util::gsl_rng_wrapper& gsl_rng,
              bnmf_algs::vector_t& cum_prob);

} // namespace util
} // namespace bnmf_algs