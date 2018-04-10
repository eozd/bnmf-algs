#pragma once

#include "defs.hpp"
#include "util/generator.hpp"
#include "util/wrappers.hpp"
#include <cstddef>
#include <utility>

namespace bnmf_algs {

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
 * \frac{1}{120x^4} - \frac{1}{252x^6} + \frac{1}{240x^8} - \frac{5}{660x^{10}} +
 * \frac{691}{32760x^{12}} - \frac{1}{12x^{14}}
 * \f]
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
 * @param x Parameter to \f$\psi(x)\f$.
 * @return \f$\psi(x)\f$.
 *
 * @see Appendix C.1 of @cite beal2003variational for a discussion of this
 * method.
 */
double psi_appr(double x) noexcept;
} // namespace util
} // namespace bnmf_algs
