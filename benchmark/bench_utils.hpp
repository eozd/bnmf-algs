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
bnmf_algs::matrixd make_matrix(long x, long y, double beg, double scale);

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
bnmf_algs::allocation_model::AllocModelParams make_params(long x, long y,
                                                          long z);
} // namespace benchmark
