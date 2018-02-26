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
 *
 * @todo Templatize the tensor elements (int, double, float, etc.)
 */
using tensor3d_t = Eigen::Tensor<double, 3, Eigen::RowMajor>;

/**
 * @brief Tensory type used to assign the results of reductions on tensors.
 *
 * @todo Templatize the tensor elements
 */
using tensor0d_t = Eigen::Tensor<double, 0, Eigen::RowMajor>;

/**
 * @brief Shape of vectors, matrices, tensors, etc.
 *
 * @tparam N Number of dimensions.
 *
 * @remark Number of dimensions must be known at compile-time.
 */
template <size_t N> using shape = std::array<size_t, N>;
} // namespace bnmf_algs
