#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace bnmf_algs {
/**
 * @brief Vector type used in the computations.
 */
template <typename Scalar>
using vector_t = Eigen::Matrix<Scalar, 1, Eigen::Dynamic, Eigen::RowMajor>;

/**
 * @brief Vector type specialization using double as Scalar value.
 */
using vectord = vector_t<double>;

/**
 * @brief Matrix type used in the computations.
 *
 * @todo Templatize the matrix elements (int, double, float, etc.)
 */
template <typename Scalar>
using matrix_t =
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/**
 * @brief Matrix type specialization using double as Scalar value.
 */
using matrixd = matrix_t<double>;

/**
 * @brief Tensor type used in the computations.
 */
template <typename Scalar, size_t N>
using tensor_t = Eigen::Tensor<Scalar, N, Eigen::RowMajor>;

/**
 * @brief Tensor type specialization using double as Scalar value.
 */
template <size_t N> using tensord = tensor_t<double, N>;

/**
 * @brief Shape of vectors, matrices, tensors, etc.
 *
 * @tparam N Number of dimensions.
 *
 * @remark Number of dimensions must be known at compile-time.
 */
template <size_t N> using shape = Eigen::array<size_t, N>;
} // namespace bnmf_algs
