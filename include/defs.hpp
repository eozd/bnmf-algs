#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace bnmf_algs {
/**
 * @brief Vector type used in the computations.
 *
 * @todo Templatize the vector elements (int, double, float, etc.)
 */
using vector_t = Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor>;
/**
 * @brief Matrix type used in the computations.
 *
 * @todo Templatize the matrix elements (int, double, float, etc.)
 */
using matrix_t =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/**
 * @brief Tensor type used in the computations.
 */
template <typename Scalar, size_t N>
using tensorx = Eigen::Tensor<Scalar, N, Eigen::RowMajor>;

/**
 * @brief Tensor type specialization using double as Scalar value.
 */
template <size_t N> using tensord = tensorx<double, N>;

/**
 * @brief Shape of vectors, matrices, tensors, etc.
 *
 * @tparam N Number of dimensions.
 *
 * @remark Number of dimensions must be known at compile-time.
 */
template <size_t N> using shape = std::array<size_t, N>;
} // namespace bnmf_algs
