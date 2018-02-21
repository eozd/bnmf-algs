#pragma once

#include <Eigen/Dense>

namespace bnmf_algs {
    /**
     * Matrix type used in the computations.
     *
     * @todo Templatize the matrix elements (int, double, float, etc.)
     */
    using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
}
