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
 * @brief Computer type that will compute the next sample from a given matrix
 * without replacement when its operator() is invoked.
 *
 * SampleOnesNoReplaceComputer will generate the next std::pair<int, int> value
 * from the previous value passed to its operator(). This operation is
 * performed in-place.
 *
 * When computing the next sample, a systematic sampling technique is used. Each
 * element and its index in the matrix to sample is stored in a heap. Then, at
 * each call to operator(), indices of the highest value is returned. Since the
 * sampling is performed without replacement, the found value is decremented and
 * pushed back to the heap.
 */
class SampleOnesNoReplaceComputer {
  public:
    /**
     * @brief Construct a new SampleOnesNoReplaceComputer.
     *
     * This method constructs the data structures that will be used during
     * sampling procedure. In particular, a max heap of elements and their
     * indices is constructed in linear time.
     *
     * @param X Matrix \f$X\f$ that will be used during sampling.
     *
     * @remark Time complexity is \f$O(N)\f$ where \f$N\f$ is the number of
     * nonzero entries in matrix parameter X.
     */
    explicit SampleOnesNoReplaceComputer(const matrixd& X);

    /**
     * @brief Function call operator that will compute the next sample without
     * replacement in-place.
     *
     * After this function exits, the object referenced by prev_val will be
     * modified to store the next sample. To read a general description of the
     * sampling procedure, refer to class documentation.
     *
     * Note that SampleOnesComputer object satisfies the
     * bnmf_algs::util::Generator and bnmf_algs::util::ComputationIterator
     * Computer interface. Hence, a sequence of samples may be generated
     * efficiently by using this Computer with bnmf_algs::util::Generator or
     * bnmf_algs::util::ComputationIterator.
     *
     * @param curr_step Current step of the sampling computation.
     * @param prev_val Previously sampled value to modify in-place.
     *
     * @remark Time complexity is \f$O(\log{N})\f$ where \f$N\f$ is the
     * number of nonzero entries of matrix parameter X that are not completely
     * sampled yet.
     */
    void operator()(size_t curr_step, std::pair<int, int>& prev_val);

    // computation variables
  private:
    using index = std::pair<int, int>;
    using val_index = std::pair<double, index>;

    std::vector<val_index> vals_indices;
    std::function<bool(const val_index&, const val_index&)> no_repl_comp;
};

/**
 * @brief Computer type that will compute the next sample from a given matrix
 * with replacement when its operator() is invoked.
 *
 * SampleOnesNoReplaceComputer will generate the next std::pair<int, int> value
 * from the previous value passed to its operator(). This operation is
 * performed in-place.
 *
 * When computing the next sample, a simple inverse CDF sampling technique is
 * used. Given matrix is thought of as a single dimensional array and its
 * entries are considered as values of a probability mass function. Cumulative
 * distribution of PMF is calculated in the constructor. At each call to
 * operator(), index of the sample is found using binary search.
 */
class SampleOnesReplaceComputer {
  public:
    /**
     * @brief Construct a new SampleOnesComputer.
     *
     * This method constructs the data structures to use during sampling
     * procedure. In particular, a cumulative probability array is calculated in
     * linear time.
     *
     * @param X Matrix \f$X\f$ that will be used during sampling.
     *
     * @remark Time complexity is \f$O(N)\f$ where \f$N\f$ is the number of
     * entries of matrix parameter X.
     */
    explicit SampleOnesReplaceComputer(const matrixd& X);

    /**
     * @brief Function call operator that will compute the next sample in-place.
     *
     * After this function exits, the object referenced by prev_val will be
     * modified to store the next sample. To read a general description of
     * sampling procedure, refer to class documentation.
     *
     * Note that SampleOnesComputer object satisfies the
     * bnmf_algs::util::Generator and bnmf_algs::util::ComputationIterator
     * Computer interface. Hence, a sequence of samples may be generated
     * efficiently by using this Computer with bnmf_algs::util::Generator or
     * bnmf_algs::util::ComputationIterator.
     *
     * @param curr_step Current step of the sampling computation.
     * @param prev_val Previously sampled value to modify in-place.
     *
     * @remark Time complexity is \f$O(N)\f$ where \f$N\f$ is the number of
     * entries of matrix parameter X.
     */
    void operator()(size_t curr_step, std::pair<int, int>& prev_val);

    // computation variables
  private:
    vectord cum_prob;
    long X_cols;
    double X_sum;
    util::gsl_rng_wrapper rnd_gen;
};
} // namespace details
namespace util {

/**
 * @brief Return a bnmf_algs::util::Generator that will generate a sequence of
 * samples from the given matrix without replacement by using
 * details::SampleOnesNoReplaceComputer as its Computer.
 *
 * This function samples the given matrix X one by one until all the entries of
 * the matrix are sampled. The returned util::Generator object will return the
 * indices of the next sample one at a time. Since the return value is a
 * generator, only a single sample is stored in the memory at a given time.
 *
 * @param X Matrix to sample one by one without replacement.
 *
 * @return A bnmf_algs::util::Generator type that can be used to generate a
 * sequence of samples from the given matrix.
 *
 * @throws std::invalid_argument if matrix \f$X\f$ has no element, if matrix
 * \f$X\f$ contains negative entries.
 */
util::Generator<std::pair<int, int>, details::SampleOnesNoReplaceComputer>
sample_ones_noreplace(const matrixd& X);

/**
 * @brief Return a bnmf_algs::util::Generator that will generate a sequence of
 * samples from the given matrix with replacement by using
 * details::SampleOnesReplaceComputer as its Computer.
 *
 * This function considers the given matrix X as a probability mass function and
 * samples random indices according to the magnitude of the entries. Entries
 * with higher values are more likely to be sampled more often. The returned
 * util::Generator object will return the indices of the next sample one at a
 * time. Since the return value is a generator, only a single sample is stored
 * in the memory at a given time.
 *
 * @param X Matrix to sample one by one with replacement.
 * @param num_samples Number of samples to generate.
 *
 * @return A bnmf_algs::util::Generator type that can be used to generate a
 * sequence of samples from the given matrix.
 *
 * @throws std::invalid_argument if matrix \f$X\f$ has no element, if matrix
 * \f$X\f$ contains negative entries.
 */
util::Generator<std::pair<int, int>, details::SampleOnesReplaceComputer>
sample_ones_replace(const matrixd& X, size_t num_samples);

/**
 * @brief Choose a random sample according to the given cumulative probability
 * distribution.
 *
 * This function chooses a sample (index) from the given cumulative probability
 * distribution. Given distribution does not have to be normalized and it is
 * not normalized inside this function. A sample is drawn uniformly from the
 * range [0, max_cum_prob] and then the corresponding index is found using
 * binary search.
 *
 * @tparam RandomIterator An iterator that satisfies RandomIterator interface.
 * This function runs in O(logn) time with RandomIterators due to the efficiency
 * in finding difference of RandomIterator types.
 *
 * @param cum_prob_begin Beginning of the cumulative probability range.
 * @param cum_prob_end End of the cumulative probability range.
 * @param gsl_rng GSL random number generator object.
 *
 * @return Iterator to the cumulative distribution value of the chosen value. If
 * no value was chosen, returns cum_prob_end.
 *
 * @remark Complexity is \f$\Theta(logn)\f$.
 *
 * @remark If all the values in the given cumulative distribution is 0, this
 * function always returns the given end iterator. This means that no values
 * were chosen.
 *
 * @remark If the given cumulative distribution is empty (begin == end), then
 * this function always returns end iterator.
 */
template <typename RandomIterator>
RandomIterator choice(RandomIterator cum_prob_begin,
                      RandomIterator cum_prob_end,
                      const bnmf_algs::util::gsl_rng_wrapper& gsl_rng) {
    // if the distribution is empty, return end iterator
    if (cum_prob_begin == cum_prob_end) {
        return cum_prob_end;
    }

    // get a uniform random number in the range [0, max_cum_prob).
    auto max_val = *(std::prev(cum_prob_end));
    double p = gsl_ran_flat(gsl_rng.get(), 0, max_val);

    // find the iterator using binary search
    return std::upper_bound(cum_prob_begin, cum_prob_end, p);
}

} // namespace util
} // namespace bnmf_algs
