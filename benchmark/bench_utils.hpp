#pragma once

#include "bnmf_algs.hpp"

/**
 * @brief benchmark namespace contains common functions and auxiliaries to be
 * used by different benchmarks.
 */
namespace benchmark {
/**
 * @brief Construct a new bnmf_algs::matrixd with given size and element
 * magnitude.
 *
 * Each element in the constructed matrix belongs to the range
 * [beg, beg + scale).
 *
 * @param x Rows of the matrix to construct.
 * @param y Columns of the matrix to construct.
 * @param beg Beginning of the entry magnitude range (included).
 * @param scale Scale of the entry magnitude range.
 *
 * @return Constructed matrix.
 */
template <typename T>
bnmf_algs::matrix_t<T> make_matrix(long x, long y, T beg, T scale) {
    using namespace bnmf_algs;

    matrix_t<T> res =
        matrix_t<T>::Random(x, y) + matrix_t<T>::Constant(x, y, beg + 1);
    res = res.array() * (scale / 2);

    return res;
}

/**
 * @brief Construct a new bnmf_algs::allocation_model::AllocModelParams with
 * parameters as given in \cite kurutmazbayesian.
 *
 * @param x Number of rows of matrix \f$X\f$.
 * @param y Number of columns of matrix \f$X\f.
 * @param z Decomposition rank.
 *
 * @return Allocation model parameters as defined in \cite kurutmazbayesian.
 */
template <typename Scalar>
bnmf_algs::alloc_model::Params<Scalar> make_params(long x, long y, long z) {
    bnmf_algs::alloc_model::Params<Scalar> params(
        100, 1, std::vector<Scalar>(x, 0.05), std::vector<Scalar>(z, 10));
    params.beta.back() = 60;

    return params;
}
} // namespace benchmark
