#pragma once

#include "cuda/memory.hpp"
#include "defs.hpp"
#include <array>

namespace bnmf_algs {
/**
 * @brief cuda namespace contains functions that operate on nvidia GPUs
 * using CUDA routines.
 */
namespace cuda {

/**
 * @brief Sum the given 3D input tensor along each of its axes and return all
 * 2D sum tensors.
 *
 * This function computes the sum of the given 3D input tensor along each
 * dimension by performing the computation on GPU using CUDA. Summing a 3D
 * \f$x \times y \times z\f$ tensor \f$S\f$ along its first axis computes a new
 * 2D tensor \f$M\f$ of shape \f$y \times z\f$ where
 *
 * \f[
 *     M_{jk} = \sum_i S_{ijk}
 * \f]
 *
 * Summing along the other axes is defined similarly. \f$i^{th}\f$ entry of the
 * output array contains the result of summing the input tensor \f$S\f$ along
 * its \f$(i + 1)^{th}\f$ axis.
 *
 * The given 3D tensor must be previously copied to the GPU. Additionally, GPU
 * memory for all sum tensors must be already allocated. cuda::DeviceMemory3D
 * and cuda::DeviceMemory2D objects provide simple APIs for these tasks.
 *
 * @tparam T Type of the entries of the input tensor.
 * @param tensor Input tensor to sum along each of its axes.
 * @param result_arr An array of sum tensors \f$(M_{y \times z}, M_{x \times z},
 * M_{x \times y})\f$.
 */
template <typename T>
void tensor_sums(const DeviceMemory3D<T>& tensor,
                 std::array<DeviceMemory2D<T>, 3>& result_arr);

} // namespace cuda
} // namespace bnmf_algs
