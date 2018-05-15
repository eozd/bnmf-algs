#pragma once

#include "defs.hpp"
#include "util/generator.hpp"
#include <cstddef>
#include <utility>

#ifdef USE_OPENMP
#include <thread>
#endif

namespace bnmf_algs {
namespace details {

// forward declaration needed to use this function in left_partition
template <typename IntIterator>
void right_partition(IntIterator begin, IntIterator end, IntIterator all_begin,
                     IntIterator all_end,
                     std::vector<std::pair<size_t, size_t>>& change_indices);

/**
 * @brief Partition the number stored at the end of the given sequence towards
 * the beginning of the sequence by computing new partitions using exactly
 * one increment and one decrement operations.
 *
 * This function partitions the number given at the end of the given sequence,
 * <b> *(end - 1) </b>, towards the beginning, <b> *begin </b>.
 * The indices of the values incremented/decremented to compute a new partition
 * is pushed to the back of the given output vector, change_indices.
 *
 * For <b> n = *(end - 1) </b> and <b> k = end - begin </b>, this
 * function stores exactly \f${n + k - 1}\choose{k - 1}\f$ different partitions
 * between begin and end iterators. Increment/decrement indices of each
 * partition is stored in the given output vector.
 *
 * @tparam Integer Integer type to use.
 * @param begin Beginning (leftmost iterator) of the sequence to store the
 * partitions of the rightmost number.
 * @param end End (rightmost past one) of the sequence to store the partitions
 * of the rightmost number. The number to partition must reside on the iterator
 * before end.
 * @param all_begin Beginning of the whole sequence that stores complete
 * partitions. This is the beginning iterator of the original vector that is
 * modified to store various partitions of the original number.
 * @param all_end End of the whole sequence that stores complete partitions.
 * @param change_indices Output vector that stores increment/decrement index
 * pairs to compute the next partition from previous one.
 */
template <typename IntIterator>
void left_partition(IntIterator begin, IntIterator end, IntIterator all_begin,
                    IntIterator all_end,
                    std::vector<std::pair<size_t, size_t>>& change_indices) {

    const auto k = end - begin;
    const auto n = *(end - 1);

    // base cases
    if (k == 1) {
        return;
    } else if (k == 2) {
        // when k = 2, simple loop is enough.
        auto& rightmost = *(end - 1);
        auto& leftmost = *(end - 2);

        const size_t rightmost_index = (end - 1) - all_begin;
        const size_t leftmost_index = (end - 2) - all_begin;

        while (rightmost > 0) {
            --rightmost;
            ++leftmost;
            change_indices.emplace_back(rightmost_index, leftmost_index);
        }
    } else {
        // compute partitions recursively
        auto& rightmost = *(end - 1);
        auto& leftmost = *begin;
        auto& leftmost_next = *(begin + 1);

        const size_t rightmost_index = (end - 1) - all_begin;
        const size_t leftmost_index = begin - all_begin;
        const size_t leftmost_next_index = (begin + 1) - all_begin;

        while (true) {
            left_partition(begin + 1, end, all_begin, all_end, change_indices);
            --leftmost_next;
            ++leftmost;
            change_indices.emplace_back(leftmost_next_index, leftmost_index);

            if (leftmost == n) {
                break;
            }

            right_partition(begin + 1, end, all_begin, all_end, change_indices);
            --rightmost;
            ++leftmost;
            change_indices.emplace_back(rightmost_index, leftmost_index);

            if (leftmost == n) {
                break;
            }
        }
    }
}

/**
 * @brief Partition the number stored at the beginning of the given sequence
 * towards the end of the sequence by computing new partitions using exactly one
 * increment and one decrement operations.
 *
 * This function partitions the number given at the beginning of the given
 * sequence,
 * <b> *begin </b>, towards the end, <b> *(end - 1) </b>.
 * The indices of the values incremented/decremented to compute a new partition
 * is pushed to the back of the given output vector, change_indices.
 *
 * For <b> n = *begin </b> and <b> k = end - begin </b>, this
 * function stores exactly \f${n + k - 1}\choose{k - 1}\f$ different partitions
 * between begin and end iterators. Increment/decrement indices of each
 * partition is stored in the given output vector.
 *
 * @tparam Integer Integer type to use.
 * @param begin Beginning (leftmost iterator) of the sequence to store the
 * partitions of the leftmost number. (*begin)
 * @param end End (rightmost past one) of the sequence to store the partitions
 * of the rightmost number.
 * @param all_begin Beginning of the whole sequence that stores complete
 * partitions. This is the beginning iterator of the original vector that is
 * modified to store various partitions of the original number.
 * @param all_end End of the whole sequence that stores complete partitions.
 * @param change_indices Output vector that stores increment/decrement index
 * pairs to compute the next partition from previous one.
 */
template <typename IntIterator>
void right_partition(IntIterator begin, IntIterator end, IntIterator all_begin,
                     IntIterator all_end,
                     std::vector<std::pair<size_t, size_t>>& change_indices) {

    const auto k = end - begin;
    const auto n = *begin;

    // base cases
    if (k == 1) {
        return;
    } else if (k == 2) {
        auto& leftmost = *begin;
        auto& rightmost = *(begin + 1);

        const size_t leftmost_index = begin - all_begin;
        const size_t rightmost_index = (begin + 1) - all_begin;

        while (leftmost > 0) {
            --leftmost;
            ++rightmost;
            change_indices.emplace_back(leftmost_index, rightmost_index);
        }
    } else {
        // compute new partitions recursively
        auto& leftmost = *begin;
        auto& rightmost = *(end - 1);
        auto& rightmost_prev = *(end - 2);

        const size_t leftmost_index = begin - all_begin;
        const size_t rightmost_index = (end - 1) - all_begin;
        const size_t rightmost_prev_index = (end - 2) - all_begin;

        while (true) {
            right_partition(begin, end - 1, all_begin, all_end, change_indices);
            --rightmost_prev;
            ++rightmost;
            change_indices.emplace_back(rightmost_prev_index, rightmost_index);

            if (rightmost == n) {
                break;
            }

            left_partition(begin, end - 1, all_begin, all_end, change_indices);
            --leftmost;
            ++rightmost;
            change_indices.emplace_back(leftmost_index, rightmost_index);

            if (rightmost == n) {
                break;
            }
        }
    }
}
} // namespace details

/**
 * @brief Namespace for general purpose utility functions to be used by all
 * bnmf_algs functions.
 */
namespace util {

/**
 * @brief Forward the selected elements of a tuple to a function and return the
 * result.
 *
 * This function takes a function pointer and a
 * (std::tuple, std::array, std::pair) and invokes the function with the chosen
 * elements of the tuple using a std::index_sequence.
 *
 * @tparam Function Function pointer type.
 * @tparam Tuple std::tuple, std::array, std::pair and so on such that
 * std::get<I>(Tuple) gets the Ith element of the pack.
 * @tparam I Scalars that can be used when indexing (integer, short, etc.)
 * @param f Function pointer to invoke.
 * @param t Tuple variant that contains the parameters to the function.
 * @param seq std::index_sequence Indices of tuple elements to forward to the
 * function.
 *
 * @return Return value of the function after being invoked with the given
 * parameters.
 *
 * @see
 * https://stackoverflow.com/questions/7858817/unpacking-a-tuple-to-call-a-matching-function-pointer
 */
template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...> seq) {
    return f(std::get<I>(t)...);
}

/**
 * @brief Forward the elements of a tuple to a function and return the result.
 *
 * This function takes a function pointer and a
 * (std::tuple, std::array, std::pair) and invokes the function with the
 * elements of the tuple. Parameter list of the function and the template types
 * of tuple elements must be the same; otherwise, a compiler error would be
 * issued.
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
 * @tparam T Type of the entries of the tensor parameter.
 * @param S 3D tensor \f$S\f$.
 *
 * @return Sparseness of \f$S\f$.
 *
 * @remark If all elements of \f$S\f$ are 0, then the function returns
 * std::numeric_limits<double>::max().
 */
template <typename T> double sparseness(const tensor_t<T, 3>& S) {
    // TODO: implement a method to unpack std::array
    auto x = static_cast<size_t>(S.dimension(0));
    auto y = static_cast<size_t>(S.dimension(1));
    auto z = static_cast<size_t>(S.dimension(2));

    T sum = 0, squared_sum = 0;
    #pragma omp parallel for schedule(static) reduction(+:sum,squared_sum)
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            for (size_t k = 0; k < z; ++k) {
                sum += S(i, j, k);
                squared_sum += S(i, j, k) * S(i, j, k);
            }
        }
    }

    if (squared_sum < std::numeric_limits<double>::epsilon()) {
        return std::numeric_limits<double>::max();
    }

    double frob_norm = std::sqrt(squared_sum);
    double axis_mult = std::sqrt(x * y * z);

    return (axis_mult - sum / frob_norm) / (axis_mult - 1);
}

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
 * @tparam Scalar Type of the tensor entries
 * @tparam N Dimension of the tensor. Must be known at compile time.
 * @param input Input tensor to normalize in-place.
 * @param axis Axis along which the tensor will be normalized.
 * @param type Normalization type. See bnmf_algs::util::NormType for details.
 * @param eps Floating point epsilon value to prevent division by 0 errors.
 */
template <typename Scalar, int N>
void normalize(tensor_t<Scalar, N>& input, size_t axis,
               NormType type = NormType::L1, double eps = 1e-50) {
    static_assert(N >= 1, "Tensor must be at least 1 dimensional");
    BNMF_ASSERT(axis < N, "Normalization axis must be less than N");

    // calculate output dimensions
    auto input_dims = input.dimensions();
    Eigen::array<long, N - 1> output_dims;
    for (size_t i = 0, j = 0; i < N; ++i) {
        if (i != axis) {
            output_dims[j] = input_dims[i];
            ++j;
        }
    }

    // initialize thread device if using openmp
#ifdef USE_OPENMP
    Eigen::ThreadPool tp(std::thread::hardware_concurrency());
    Eigen::ThreadPoolDevice thread_dev(&tp,
                                       std::thread::hardware_concurrency());
#endif

    // calculate norms
    tensor_t<Scalar, N - 1> norms(output_dims);
    shape<1> reduction_dim{axis};
    switch (type) {
    case NormType::L1:
#ifdef USE_OPENMP
        norms.device(thread_dev) = input.abs().sum(reduction_dim);
#else
        norms = input.abs().sum(reduction_dim);
#endif
        break;
    case NormType::L2:
#ifdef USE_OPENMP
        norms.device(thread_dev) = input.square().sum(reduction_dim);
#else
        norms = input.square().sum(reduction_dim);
#endif
        break;
    case NormType::Inf:
#ifdef USE_OPENMP
        norms.device(thread_dev) = input.abs().maximum(reduction_dim);
#else
        norms = input.abs().maximum(reduction_dim);
#endif
        break;
    }

    // add epsilon to norms values to prevent division by zeros
    Scalar* norms_begin = norms.data();
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < norms.size(); ++i) {
        norms_begin[i] += eps;
    }

    // divide along each slice
    int axis_dims = input.dimensions()[axis];
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < axis_dims; ++i) {
        input.chip(i, axis) /= norms;
    }
}

/**
 * @brief Return a normalized copy of a tensor along the given axis.
 *
 * @tparam Scalar Type of the tensor entries
 * @tparam N Dimension of the tensor. Must be known at compile time.
 * @param input Input tensor to normalize.
 * @param axis Axis along which the tensor will be normalized.
 * @param type Normalization type. See bnmf_algs::util::NormType for details.
 * @param eps Floating point epsilon value to prevent division by 0 errors.
 *
 * @return Normalized copy of the input tensor along the given axis
 */
template <typename Scalar, int N>
tensor_t<Scalar, N> normalized(const tensor_t<Scalar, N>& input, size_t axis,
                               NormType type = NormType::L1,
                               double eps = 1e-50) {
    tensor_t<Scalar, N> out = input;
    normalize(out, axis, type, eps);
    return out;
}

/**
 * @brief Calculate digamma function approximately using the first 8 terms of
 * the asymptotic expansion of digamma function.
 *
 * Digamma function is defined as
 *
 * \f[
 *     \psi(x) = \frac{\Gamma'(x)}{\Gamma(x)}
 * \f]
 *
 * This function computes an approximation for \f$\psi(x)\f$ using the below
 * formula:
 *
 * \f[
 *     \psi(x) \approx \ln(x) - \frac{1}{2x} - \frac{1}{12x^2} +
 * \frac{1}{120x^4} - \frac{1}{252x^6} + \frac{1}{240x^8} - \frac{5}{660x^{10}}
 * + \frac{691}{32760x^{12}} - \frac{1}{12x^{14}} \f]
 *
 * This approximation is more accurate for larger values of \f$x\f$. When
 * computing \f$\psi(x)\f$ for \f$x < 6\f$, the below recurrence relation is
 * used to shift the \f$x\f$ value to use in the approximation formula to a
 * value greater than \f$6\f$:
 *
 * \f[
 *     \psi(x + 1) = \frac{1}{x} + \psi(x)
 * \f]
 *
 * @tparam Real A real scalar value such as double and float.
 * @param x Parameter to \f$\psi(x)\f$.
 * @return \f$\psi(x)\f$.
 *
 * @see Appendix C.1 of @cite beal2003variational for a discussion of this
 * method.
 */
template <typename Real> Real psi_appr(Real x) {
    Real extra = 0;

    // write each case separately to minimize number of divisions as much as
    // possible
    if (x < 1) {
        const Real a = x + 1;
        const Real b = x + 2;
        const Real c = x + 3;
        const Real d = x + 4;
        const Real e = x + 5;
        const Real ab = a * b;
        const Real cd = c * d;
        const Real ex = e * x;

        extra = ((a + b) * cd * ex + ab * d * ex + ab * c * ex + ab * cd * x +
                 ab * cd * e) /
                (ab * cd * ex);
        x += 6;
    } else if (x < 2) {
        const Real a = x + 1;
        const Real b = x + 2;
        const Real c = x + 3;
        const Real d = x + 4;
        const Real ab = a * b;
        const Real cd = c * d;
        const Real dx = d * x;

        extra =
            ((a + b) * c * dx + ab * dx + ab * c * x + ab * cd) / (ab * cd * x);
        x += 5;
    } else if (x < 3) {
        const Real a = x + 1;
        const Real b = x + 2;
        const Real c = x + 3;
        const Real ab = a * b;
        const Real cx = c * x;

        extra = ((a + b) * cx + (c + x) * ab) / (ab * cx);
        x += 4;
    } else if (x < 4) {
        const Real a = x + 1;
        const Real b = x + 2;
        const Real ab = a * b;

        extra = ((a + b) * x + ab) / (ab * x);
        x += 3;
    } else if (x < 5) {
        const Real a = x + 1;

        extra = (a + x) / (a * x);
        x += 2;
    } else if (x < 6) {
        extra = 1 / x;
        x += 1;
    }

    Real x2 = x * x;
    Real x4 = x2 * x2;
    Real x6 = x4 * x2;
    Real x8 = x6 * x2;
    Real x10 = x8 * x2;
    Real x12 = x10 * x2;
    Real x13 = x12 * x;
    Real x14 = x13 * x;

    // write the result of the formula simplified using symbolic calculation
    // to minimize the number of divisions
    Real res =
        std::log(x) + (-360360 * x13 - 60060 * x12 + 6006 * x10 - 2860 * x8 +
                       3003 * x6 - 5460 * x4 + 15202 * x2 - 60060) /
                          (720720 * x14);

    return res - extra;
}

/**
 * @brief Compute the sequence of indices to be incremented and decremented to
 * generate all partitions of number n into k bins.
 *
 * This function computes an index pair for each k length partition of number n.
 * The pair holds the indices of the value to decrement by 1 and increment by 1,
 * respectively. Starting from the initial partition, \f$(n, 0, \dots, 0)\f$,
 * all k length partitions of number n can be generated by iteratively applying
 * the decrement and increment operators on the returned indices. For example,
 * this fuction returns the following index sequence when called with \f$n = 2,
 * k = 3\f$.
 *
 * \f[
 *     (0, 1)
 *     (0, 1)
 *     (1, 2)
 *     (1, 0)
 *     (0, 2)
 * \f]
 *
 * Starting from initial partition, \f$(2, 0, 0)\f$, if we decrement the value
 * at the 0th index and increment the value at 1st index, we generate all
 * partitions:
 *
 * \f[
 *     (2, 0, 0)
 *     (1, 1, 0)
 *     (0, 2, 0)
 *     (0, 1, 1)
 *     (1, 0, 1)
 *     (0, 0, 2)
 * \f]
 *
 * @tparam Integer Integer type to use.
 * @param n Number to divide into k partitions.
 * @param k Number of partitions.
 * @return std::vector containing indices to decrement and increment as
 * std::pair types. std::pair::first contains the index of element to decrement,
 * and std::pair::second contains the index of element to increment.
 *
 * @throws assertion error if k is less than or equal to 0, if n is less than or
 * equal to 0.
 */
template <typename Integer>
std::vector<std::pair<size_t, size_t>> partition_change_indices(Integer n,
                                                                Integer k) {
    BNMF_ASSERT(k > 0, "k must be greater than 0 in util::all_partitions");
    BNMF_ASSERT(n > 0, "n must be greater than 0 in util::all_partitions");

    // initial partition
    std::vector<Integer> vec(k, 0);
    vec[0] = n;

    // partition towards right
    std::vector<std::pair<size_t, size_t>> result;
    details::right_partition(vec.begin(), vec.end(), vec.begin(), vec.end(),
                             result);

    return result;
}

/**
 * @brief Compute all possible k length partitions of given number n.
 *
 * This function computes all possible k length partitions of number n. There
 * are exactly {N + k - 1}\choose{k - 1} many k length partitions of number n.
 *
 * The returned partition sequence has a special property: All consecutive
 * partitions differ exactly by 1, i.e. <b>only one of the bins are decremented
 * by 1 and only one of the bins are incremented by 1</b>. All the other bins
 * remain unchanged.
 *
 *
 * When called with \f$n = 2, k = 3\f$, this function returns:
 *
 * \f[
 *     (2, 0, 0)
 *     (1, 1, 0)
 *     (0, 2, 0)
 *     (0, 1, 1)
 *     (1, 0, 1)
 *     (0, 0, 2)
 * \f]
 *
 * in the exacty same order.
 *
 * @tparam Integer Integer type to use.
 * @param n Number to divide into k partitions.
 * @param k Number of partitions.
 * @return std::vector containing all k length partitions as std::vector types.
 * All the partition vectors are of length k.
 *
 * @throws assertion error if k is less than or equal to 0, if n is less than or
 * equal to 0.
 */
template <typename Integer>
std::vector<std::vector<Integer>> all_partitions(Integer n, Integer k) {
    // get indices to increment/decrement
    auto part_change_vec = partition_change_indices(n, k);

    // compute partitions iteratively by applying increment/decrement operations
    std::vector<std::vector<Integer>> result;
    std::vector<Integer> partition(k, 0);
    partition[0] = n;
    result.push_back(partition);
    size_t decr_idx, incr_idx;
    for (const auto& change_idx : part_change_vec) {
        std::tie(decr_idx, incr_idx) = change_idx;
        --partition[decr_idx];
        ++partition[incr_idx];
        result.push_back(partition);
    }

    return result;
}

} // namespace util
} // namespace bnmf_algs
