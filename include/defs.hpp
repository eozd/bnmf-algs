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
    using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    /**
     * @brief Tensor type used in the computations.
     *
     * @todo Templatize the tensor elements (int, double, float, etc.)
     */
    using tensor3d_t = Eigen::Tensor<double, 3, Eigen::RowMajor>;
}
