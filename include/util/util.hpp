#pragma once

#include "defs.hpp"
#include "util/wrappers.hpp"
#include "util/generator.hpp"
#include <cstddef>
#include <utility>

namespace bnmf_algs {
/**
 * @brief Namespace for general purpose utility functions to be used by all
 * bnmf_algs functions.
 */

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
 * @brief Forward the selected elements of a tuple to a function and return the
 * result.
 *
 * This function takes a function pointer and a
 * (std::tuple/std::array/std::pair) and invokes the function with the chosen
 * elements of the tuple using a std::index_sequence.
 *
 * @tparam Function Function pointer type.
 * @tparam Tuple std::tuple, std::array, std::pair and so on such that
 * std::get<I>(Tuple) gets the Ith element of the pack.
 * @tparam A scalar that can be used when indexing (integer, short, etc.)
 * @param f Function pointer to invoke.
 * @param t Tuple variant that contains the parameters to the function.
 * @param std::index_sequence Indices of tuple elements to forward to the
 * function.
 *
 * @return Return value of the function after being invoked with the given
 * parameters.
 *
 * @see
 * https://stackoverflow.com/questions/7858817/unpacking-a-tuple-to-call-a-matching-function-pointer
 */
template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>) {
    return f(std::get<I>(t)...);
}

/**
 * @brief Forward the elements of a tuple to a function and return the result.
 *
 * This function takes a function pointer and a
 * (std::tuple/std::array/std::pair) and invokes the function with the elements
 * of the tuple. Parameter list of the function and the template types of tuple
 * elements must be the same; otherwise, a compiler error would be issued.
 *
 * @tparam Function Function pointer type.
 * @tparam Tuple std::tuple, std::array, std::pair and so on such that
 * std::get<I>(Tuple) gets the Ith element of the pack.
 * @param f Function pointer to invoke.
 * @param t Tuple variant that contains the parameters to the function.
 *
 * @return Return value of the function after being invoked with the given
 * parameters.
 *
 * @see
 * https://stackoverflow.com/questions/7858817/unpacking-a-tuple-to-call-a-matching-function-pointer
 */
template <typename Function, typename Tuple> auto call(Function f, Tuple t) {
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return call(f, t, std::make_index_sequence<size>{});
}

/**
 * @brief Compute the sparseness of the given 3D tensor.
 *
 * Sparseness of a 3D tensor \f$S_{x \times y \times z}\f$ is defined as
 *
 * \f[
 * \frac{\sqrt{xyz} - S_{+++}/\|S\|_F}{\sqrt{xyz} - 1}
 * \f]
 *
 * where \f$\|S\|_F = \sqrt{\sum_{ijk}(S_{ijk})^2}\f$ is the Frobenius norm of
 * tensor \f$S\f$ and \f$S_{+++} = \sum_{ijk}S_{ijk}\f$ is the sum of elements
 * of \f$S\f$.
 *
 * @param S 3D tensor \f$S\f$.
 *
 * @return Sparseness of \f$S\f$.
 *
 * @remark If all elements of \f$S\f$ are 0, then the function returns
 * std::numeric_limits<double>::max().
 */
double sparseness(const tensord<3>& S);

/**
 * @brief Type of the norm to be used by bnmf_algs::util::normalize and
 * bnmf_algs::util::normalized.
 */
enum class NormType {
    L1, ///< Normalize using \f$L_1\f$ norm \f$\left(\|x\|_1 = \sum_i
        ///  |x_i|\right)\f$
    L2, ///< Normalize using \f$L_2\f$ norm \f$\left(\|x\|_2 = \sqrt{\sum_i
        ///  x_i^2}\right)\f$
    Inf ///< Normalize using \f$L_{\infty}\f$ norm \f$\left(\|x\|_{\infty} =
        /// \max\{|x_1|, \dots, |x_n|\}\right)\f$
};

/**
 * @brief Normalize a tensor in-place along the given axis.
 *
 * @tparam N Dimension of the tensor. Must be known at compile time.
 * @param input Input tensor to normalize in-place.
 * @param axis Axis along which the tensor will be normalized.
 * @param type Normalization type. See bnmf_algs::util::NormType for details.
 * @param eps Floating point epsilon value to prevent division by 0 errors.
 */
template <int N>
void normalize(tensord<N>& input, size_t axis, NormType type = NormType::L1,
               double eps = 1e-50) {
    static_assert(N >= 1, "Tensor must be at least 1 dimensional");
    if (axis >= N) {
        throw std::invalid_argument("Normalization axis must be less than N");
    }

    tensord<N - 1> norms;
    shape<1> reduction_dim{axis};
    switch (type) {
    case NormType::L1:
        norms = input.abs().sum(reduction_dim);
        break;
    case NormType::L2:
        norms = input.square().sum(reduction_dim);
        break;
    case NormType::Inf:
        norms = input.abs().maximum(reduction_dim);
        break;
    }
    {
        tensord<N - 1> eps_tensor(norms.dimensions());
        eps_tensor.setConstant(eps);
        norms += eps_tensor;
    }

    int axis_dims = input.dimensions()[axis];
    for (int i = 0; i < axis_dims; ++i) {
        input.chip(i, axis) /= norms;
    }
}

/**
 * @brief Return a normalized copy of a tensor along the given axis.
 *
 * @tparam N Dimension of the tensor. Must be known at compile time.
 * @param input Input tensor to normalize.
 * @param axis Axis along which the tensor will be normalized.
 * @param type Normalization type. See bnmf_algs::util::NormType for details.
 * @param eps Floating point epsilon value to prevent division by 0 errors.
 *
 * @return Normalized copy of the input tensor along the given axis
 */
template <int N>
tensord<N> normalized(const tensord<N>& input, size_t axis,
                      NormType type = NormType::L1, double eps = 1e-50) {
    tensord<N> out = input;
    normalize(out, axis, type);
    return out;
}

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
} // namespace util
} // namespace bnmf_algs
