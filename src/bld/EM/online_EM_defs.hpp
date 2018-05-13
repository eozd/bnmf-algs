#pragma once

#include "defs.hpp"

namespace bnmf_algs {
namespace bld {

/**
 * @brief Structure holding the results of EM procedures.
 *
 * @tparam T Type of the matrix/tensor entries.
 */
template <typename T> struct EMResult {
  public:
    /**
     * @brief Sum of the hidden tensor \f$S\f$ along its first dimension, i.e.
     * \f$S_{+jk}\f$.
     */
    matrix_t<T> S_pjk;
    /**
     * @brief Sum of the hidden tensor \f$S\f$ along its second dimension, i.e.
     * \f$S_{i+k}\f$.
     */
    matrix_t<T> S_ipk;
    /**
     * @brief Completed version of the incomplete matrix given as input to an
     * EM algorithm.
     *
     * EM algorithms take incomplete matrices (NaN entries are not known), and
     * find the optimal values for those empty values. This is the matrix that
     * contains the completed values for those entries.
     */
    matrix_t<T> X_full;
    /**
     * @brief Matrix whose \f$(i, j)^{th}\f$ entry contains \f$\log{W_{ij}}\f$.
     */
    matrix_t<double> logW;
    /**
     * @brief Matrix whose \f$(i, j)^{th}\f$ entry contains \f$log{H_{ij}}\f$.
     */
    matrix_t<double> logH;
    /**
     * @brief Vector containing EM bound computed after every iteration.
     */
    vector_t<double> log_PS;

  public:
    /**
     * @brief Default constructor.
     *
     * Default constructor constructs every matrix/vector as empty.
     */
    EMResult() = default;

    /**
     * @brief Initialization constructor.
     *
     * Every element is move initialized with the given value.
     */
    EMResult(matrix_t<T> S_pjk, matrix_t<T> S_ipk, matrix_t<T> X_full,
             matrix_t<T> logW, matrix_t<T> logH, vector_t<T> EM_bound)
        : S_pjk(std::move(S_pjk)), S_ipk(std::move(S_ipk)),
          X_full(std::move(X_full)), logW(std::move(logW)),
          logH(std::move(logH)), log_PS(std::move(log_PS)) {}
};
} // namespace bld
} // namespace bnmf_algs
