#pragma once

#include "defs.hpp"
#include "util/generator.hpp"
#include <algorithm>
#include <tuple>

namespace bnmf_algs {
namespace util {
/**
 * @brief Choose a random sample according to the given cumulative
 * probability distribution.
 *
 * This function chooses a sample (index) from the given cumulative
 * probability distribution. Given distribution does not have to be
 * normalized and it is not normalized inside this function. A sample is
 * drawn uniformly from the range [0, max_cum_prob] and then the
 * corresponding index is found using binary search.
 *
 * @tparam RandomIterator An iterator that satisfies RandomIterator
 * interface. This function runs in O(logn) time with RandomIterators due to
 * the efficiency in finding difference of RandomIterator types.
 *
 * @param cum_prob_begin Beginning of the cumulative probability range.
 * @param cum_prob_end End of the cumulative probability range.
 * @param gsl_rng GSL random number generator object.
 *
 * @return Iterator to the cumulative distribution value of the chosen
 * value. If no value was chosen, returns cum_prob_end.
 *
 * @remark Complexity is \f$\Theta(logn)\f$.
 *
 * @remark If all the values in the given cumulative distribution is 0, this
 * function always returns the given end iterator. This means that no values
 * were chosen.
 *
 * @remark If the given cumulative distribution is empty (begin == end),
 * then this function always returns end iterator.
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

template <typename Real, typename Integer>
void multinomial_mode(Integer num_trials, const vector_t<Real>& prob,
                      vector_t<Integer>& count, double eps = 1e-50) {
    BNMF_ASSERT(prob.cols() == count.cols(),
                "Number of event probabilities and counts differ in "
                "util::multinomial_mode");
    const auto num_events = prob.cols();

    vector_t<Real> count_real;
    vector_t<Real> diff;
    {
        vector_t<Real> freq = (num_trials + 0.5 * num_events) * prob.array();
        count_real = freq.array().floor();
        diff = freq - count_real;
        count = count_real.template cast<Integer>();
    }
    const Integer total_count = count.sum();

    if (total_count == num_trials) {
        return;
    }

    // normalized differences will be used as a heap for the greedy
    // algorithm
    std::vector<std::pair<int, Real>> diff_normalized(num_events);
    {
        // compute normalized differences
        vector_t<Real> diff_norm_vec;
        if (total_count < num_trials) {
            diff_norm_vec = (1 - diff.array()) / (count_real.array() + 1);
        } else if (total_count > num_trials) {
            diff_norm_vec = diff.array() / (count_real.array() + eps);
        }

        // store differences and their indices
        for (int i = 0; i < num_events; ++i) {
            diff_normalized[i].first = i;
            diff_normalized[i].second = diff_norm_vec(i);
        }
    }

    // min heap ordering function
    auto ordering = [](const std::pair<int, Real>& elem_left,
                       const std::pair<int, Real>& elem_right) {
        // min heap
        return elem_left.second > elem_right.second;
    };
    std::make_heap(diff_normalized.begin(), diff_normalized.end(), ordering);

    // distribute the remaining balls
    for (Integer i = total_count; i < num_trials; ++i) {
        // get the smallest normalized difference and its index
        std::pop_heap(diff_normalized.begin(), diff_normalized.end(), ordering);
        int min_idx = diff_normalized.back().first;

        // update
        ++count(min_idx);
        --diff(min_idx);
        diff_normalized.back().second =
            (1 - diff(min_idx)) / (count(min_idx) + 1);

        // push back to the heap
        std::push_heap(diff_normalized.begin(), diff_normalized.end(),
                       ordering);
    }

    // remove excess balls
    for (Integer i = total_count; i > num_trials; --i) {
        // get the smallest normalized difference and its index
        std::pop_heap(diff_normalized.begin(), diff_normalized.end(), ordering);
        int min_idx = diff_normalized.back().first;

        // update
        --count(min_idx);
        ++diff(min_idx);
        diff_normalized.back().second = diff(min_idx) / (count(min_idx) + eps);

        // push back to the heap
        std::push_heap(diff_normalized.begin(), diff_normalized.end(),
                       ordering);
    }
}
} // namespace util

/**
 * @brief Namespace that contains types and functions for internal
 * computations.
 *
 * Functions and classes in this namespace are not meant for direct
 * usage by the users of bnmf-algs library; hence, behaviours of some
 * functions may not be as expected. Additionally, some of these
 * functions and types are not directly tested, and their behaviour is
 * verified through correct behaviour of the API functions and types
 * using them.
 */
namespace details {

/**
 * @brief Computer type that will compute the next sample from a given
 * matrix without replacement when its operator() is invoked.
 *
 * SampleOnesNoReplaceComputer will generate the next std::pair<int,
 * int> value from the previous value passed to its operator(). This
 * operation is performed in-place.
 *
 * When computing the next sample, a systematic sampling technique is
 * used. Each entry of the input matrix, \f$X_{ij}\f$, is sampled
 * \f$\lceil X_{ij} \rceil\f$ many times. This can be thought of as
 * sampling each entry one-by-one. We call the implemented procedure
 * systematic because, conceptually, samples from every \f$X_{ij}\f$
 * are distributed on a timeline. Afterwards, all sample timelines are
 * merged and the sample closest to our current timestep is chosen.
 * This is easily implemented using a minimum heap ordered according to
 * timestamp of each sample.
 *
 * The above described procedure is conceptual. To decrease the space,
 * and therefore time, complexity of the heap ordering procedures,
 * timestamp of the next sample is computed just after the current
 * sample is drawn. Therefore, the maximum number of elements in the
 * heap is exactly \f$xy\f$ where \f$x\f$ and \f$y\f$ are the
 * dimensions of the input matrix. This number gradually decreases as
 * matrix entries become less than or equal to 0 and get discarded from
 * the heap.
 *
 * Sample timestamps are drawn from Beta distribution,
 * \f$\mathcal{B}(\alpha, \beta)\f$. Here, \f$\alpha = X_{ij} - t +
 * 1\f$ and \f$\beta = 1\f$ where \f$t\f$ is the number of samples
 * drawn from entry \f$X_{ij}\f$.
 *
 * @tparam Scalar Type of the matrix entries that will be sampled.
 */
template <typename Scalar> class SampleOnesNoReplaceComputer {
  public:
    /**
     * @brief Construct a new SampleOnesNoReplaceComputer.
     *
     * This method constructs the data structures that will be used
     * during sampling procedure. In particular, a min heap of elements
     * and their indices is constructed in time linear in the number of
     * matrix entries.
     *
     * @param X Matrix \f$X\f$ that will be used during sampling.
     *
     * @remark Time complexity is \f$O(N)\f$ where \f$N\f$ is the
     * number of elements of X.
     */
    explicit SampleOnesNoReplaceComputer(const matrix_t<Scalar>& X)
        : m_heap(),
          no_repl_comp([](const HeapElem& left, const HeapElem& right) {
              // min heap with respect to timestamp
              return left.timestamp >= right.timestamp;
          }),
          rnd_gen(gsl_rng_alloc(gsl_rng_taus), gsl_rng_free) {

        for (int i = 0; i < X.rows(); ++i) {
            for (int j = 0; j < X.cols(); ++j) {
                Scalar entry = X(i, j);
                if (entry > 0) {
                    double timestamp =
                        gsl_ran_beta(rnd_gen.get(), entry + 1, 1);

                    m_heap.emplace_back(timestamp, entry, std::make_pair(i, j));
                }
            }
        }
        std::make_heap(m_heap.begin(), m_heap.end(), no_repl_comp);
    }
    /**
     * @brief Function call operator that will compute the next sample
     * without replacement in-place.
     *
     * After this function exits, the object referenced by prev_val
     * will be modified to store the next sample. To read a general
     * description of the sampling procedure, refer to class
     * documentation.
     *
     * Note that SampleOnesComputer object satisfies the
     * bnmf_algs::util::Generator and
     * bnmf_algs::util::ComputationIterator Computer interface. Hence,
     * a sequence of samples may be generated efficiently by using this
     * Computer with bnmf_algs::util::Generator or
     * bnmf_algs::util::ComputationIterator.
     *
     * @param curr_step Current step of the sampling computation.
     * @param prev_val Previously sampled value to modify in-place.
     *
     * @remark Time complexity is \f$O(\log{N})\f$ where \f$N\f$ is the
     * number of nonzero entries of matrix parameter X that are not
     * completely sampled yet.
     */
    void operator()(size_t curr_step, std::pair<int, int>& prev_val) {
        // min heap sampling. Root contains the entry with the smallest
        // beta value
        if (m_heap.empty()) {
            return;
        }
        std::pop_heap(m_heap.begin(), m_heap.end(), no_repl_comp);

        auto& heap_entry = m_heap.back();
        int ii, jj;
        std::tie(ii, jj) = heap_entry.idx;

        // decrement entry value and update beta value
        --heap_entry.matrix_entry;
        if (heap_entry.matrix_entry <= 0) {
            m_heap.pop_back();
        } else {
            heap_entry.timestamp +=
                gsl_ran_beta(rnd_gen.get(), heap_entry.matrix_entry + 1, 1);
            std::push_heap(m_heap.begin(), m_heap.end(), no_repl_comp);
        }

        // store indices of the sample
        prev_val.first = ii;
        prev_val.second = jj;
    }

  private:
    /**
     * @brief A single entry of m_heap.
     */
    struct HeapElem {
        HeapElem(double timestamp, Scalar matrix_entry, std::pair<int, int> idx)
            : timestamp(timestamp), matrix_entry(matrix_entry),
              idx(std::move(idx)) {}

        /**
         * @brief Timestamp of the sample.
         */
        double timestamp;

        /**
         * @brief \f$X_{ij}\f$ where \f$X\f$ is the matrix to be
         * sampled.
         */
        Scalar matrix_entry;

        /**
         * @brief Index of \f$X_{ij}\f$ entry, i.e. (i, j).
         */
        std::pair<int, int> idx;
    };

  private:
    /**
     * @brief Heap to be used during systematic sampling.
     */
    std::vector<HeapElem> m_heap;

    /**
     * @brief Comparator function to determine the ordering of heap
     * elements.
     */
    std::function<bool(const HeapElem&, const HeapElem&)> no_repl_comp;

    /**
     * @brief GSL rng object that will be used to draw samples.
     */
    util::gsl_rng_wrapper rnd_gen;
};

/**
 * @brief Computer type that will compute the next sample from a given
 * matrix with replacement when its operator() is invoked.
 *
 * SampleOnesNoReplaceComputer will generate the next std::pair<int,
 * int> value from the previous value passed to its operator(). This
 * operation is performed in-place.
 *
 * When computing the next sample, a simple inverse CDF sampling
 * technique is used. Given matrix is thought of as a single
 * dimensional array and its entries are considered as values of a
 * probability mass function. Cumulative distribution of PMF is
 * calculated in the constructor. At each call to operator(), index of
 * the sample is found using binary search.
 *
 * @tparam Scalar Type of the matrix entries that will be sampled.
 */
template <typename Scalar> class SampleOnesReplaceComputer {
  public:
    /**
     * @brief Construct a new SampleOnesComputer.
     *
     * This method constructs the data structures to use during
     * sampling procedure. In particular, a cumulative probability
     * array is calculated in linear time.
     *
     * @param X Matrix \f$X\f$ that will be used during sampling.
     *
     * @remark Time complexity is \f$O(N)\f$ where \f$N\f$ is the
     * number of entries of matrix parameter X.
     */
    explicit SampleOnesReplaceComputer(const matrix_t<Scalar>& X)
        : cum_prob(), X_cols(X.cols()), X_sum(X.array().sum()),
          rnd_gen(util::gsl_rng_wrapper(gsl_rng_alloc(gsl_rng_taus),
                                        gsl_rng_free)) {

        cum_prob = vector_t<Scalar>(X.rows() * X.cols());
        auto X_arr = X.array();
        cum_prob(0) = X_arr(0);
        for (int i = 1; i < X_arr.size(); ++i) {
            cum_prob(i) = cum_prob(i - 1) + X_arr(i);
        }
    }

    /**
     * @brief Function call operator that will compute the next sample
     * in-place.
     *
     * After this function exits, the object referenced by prev_val
     * will be modified to store the next sample. To read a general
     * description of sampling procedure, refer to class documentation.
     *
     * Note that SampleOnesComputer object satisfies the
     * bnmf_algs::util::Generator and
     * bnmf_algs::util::ComputationIterator Computer interface. Hence,
     * a sequence of samples may be generated efficiently by using this
     * Computer with bnmf_algs::util::Generator or
     * bnmf_algs::util::ComputationIterator.
     *
     * @param curr_step Current step of the sampling computation.
     * @param prev_val Previously sampled value to modify in-place.
     *
     * @remark Time complexity is \f$O(N)\f$ where \f$N\f$ is the
     * number of entries of matrix parameter X.
     */
    void operator()(size_t curr_step, std::pair<int, int>& prev_val) {
        // inverse CDF sampling
        Scalar* begin = cum_prob.data();
        Scalar* end = cum_prob.data() + cum_prob.cols();
        auto it = util::choice(begin, end, rnd_gen);
        long m = it - begin;

        prev_val.first = static_cast<int>(m / X_cols);
        prev_val.second = static_cast<int>(m % X_cols);
    }

    // computation variables
  private:
    vector_t<Scalar> cum_prob;
    long X_cols;
    double X_sum;
    util::gsl_rng_wrapper rnd_gen;
};

template <typename T> void check_sample_ones_params(const matrix_t<T>& X) {
    BNMF_ASSERT(X.array().size() != 0,
                "Matrix X must have at least one element");
    BNMF_ASSERT((X.array() >= 0).all(), "Matrix X must be nonnegative");
    BNMF_ASSERT((X.array() > std::numeric_limits<double>::epsilon()).any(),
                "Matrix X must have at least one nonzero element");
}
} // namespace details

namespace util {
/**
 * @brief Return a bnmf_algs::util::Generator that will generate a
 * sequence of samples from the given matrix without replacement by
 * using details::SampleOnesNoReplaceComputer as its Computer.
 *
 * This function samples the given matrix X one by one until all the
 * entries of the matrix are sampled. The returned util::Generator
 * object will return the indices of the next sample one at a time.
 * Since the return value is a generator, only a single sample is
 * stored in the memory at a given time. To learn the details of how
 * the samples are generated, refer to
 * details::SampleOnesNoReplaceComputer documentation.
 *
 * @param X Matrix to sample one by one without replacement.
 *
 * @return A bnmf_algs::util::Generator type that can be used to
 * generate a sequence of samples from the given matrix.
 *
 * @throws std::invalid_argument if matrix \f$X\f$ has no element, if
 * matrix \f$X\f$ contains negative entries.
 */
template <typename T>
util::Generator<std::pair<int, int>, details::SampleOnesNoReplaceComputer<T>>
sample_ones_noreplace(const matrix_t<T>& X) {
    details::check_sample_ones_params(X);

    auto num_samples = static_cast<size_t>(X.array().sum());
    // don't know the initial value
    std::pair<int, int> init_val;
    util::Generator<std::pair<int, int>,
                    details::SampleOnesNoReplaceComputer<T>>
        gen(init_val, num_samples + 1,
            details::SampleOnesNoReplaceComputer<T>(X));

    // get rid of the first empty value
    ++gen.begin();

    return gen;
}

/**
 * @brief Return a bnmf_algs::util::Generator that will generate a
 * sequence of samples from the given matrix with replacement by using
 * details::SampleOnesReplaceComputer as its Computer.
 *
 * This function considers the given matrix X as a probability mass
 * function and samples random indices according to the magnitude of
 * the entries. Entries with higher values are more likely to be
 * sampled more often. The returned util::Generator object will return
 * the indices of the next sample one at a time. Since the return value
 * is a generator, only a single sample is stored in the memory at a
 * given time.
 *
 * @param X Matrix to sample one by one with replacement.
 * @param num_samples Number of samples to generate.
 *
 * @return A bnmf_algs::util::Generator type that can be used to
 * generate a sequence of samples from the given matrix.
 *
 * @throws std::invalid_argument if matrix \f$X\f$ has no element, if
 * matrix \f$X\f$ contains negative entries.
 */
template <typename T>
util::Generator<std::pair<int, int>, details::SampleOnesReplaceComputer<T>>
sample_ones_replace(const matrix_t<T>& X, size_t num_samples) {
    details::check_sample_ones_params(X);

    // don't know the initial value
    std::pair<int, int> init_val;
    util::Generator<std::pair<int, int>, details::SampleOnesReplaceComputer<T>>
        gen(init_val, num_samples + 1,
            details::SampleOnesReplaceComputer<T>(X));

    // get rid of the first empty value
    ++gen.begin();

    return gen;
}
} // namespace util
} // namespace bnmf_algs
