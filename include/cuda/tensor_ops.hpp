#pragma once

#include "defs.hpp"

namespace bnmf_algs {
/**
 * @brief cuda namespace contains functions that operate on nvidia GPUs
 * using CUDA routines.
 */
namespace cuda {

/**
 * @brief Initialize CUDA runtime.
 *
 * This function initializes CUDA runtime so that future CUDA library calls
 * don't incur the cost of initializing the library.
 */
void init();

/**
 * @brief Sum the given 3D input tensor along each of its axes and return all
 * 2D sum tensors.
 *
 * This function computes the sum of the given 3D input tensor along each
 * dimension by performing the computation on GPU using CUDA. Summing a 3D
 * \f$x \times y \times z\f$ tensor \f$S\f$ along its first axis creates a new
 * 2D tensor \f$M\f$ of shape \f$y \times z\f$ where
 *
 * \f[
 *     M_{jk} = \sum_i S_{ijk}
 * \f]
 *
 * Summing along the other axes is defined similarly. \f$i^{th}\f$ entry of the
 * returned array contains the result of summing the input tensor \f$S\f$ along
 * its \f$(i + 1)^{th}\f$ axis.
 *
 * Since sending and receiving large tensors to/from GPU takes a long time, this
 * function sends the input tensor only once to the GPU. Each sum tensor is
 * deallocated after it is computed to make space for the other sum tensors.
 * Therefore, the total consumed space on GPU is \f$xyz + \max\{xy, xz,
 * yz\}\f$ number of elements.
 *
 * @param tensor Input tensor to sum along each of its axes.
 * @return An array of sum tensors \f$(M_{y \times z}, M_{x \times z}, M_{x
 * \times y})\f$.
 */
std::array<bnmf_algs::tensord<2>, 3>
tensor_sums(const bnmf_algs::tensord<3>& tensor);

} // namespace cuda
} // namespace bnmf_algs
