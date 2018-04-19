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
 * @param An array of sum tensors \f$(M_{y \times z}, M_{x \times z}, M_{x
 * \times y})\f$.
 */
template <typename T>
void tensor_sums(const DeviceMemory3D<T>& tensor,
                 std::array<DeviceMemory2D<T>, 3>& result_arr);

/**
 * @brief Apply cuda::kernel::psi_appr function to every element in the given
 * 1D sequence on the GPU.
 *
 * This function makes the following update to every \f$x\f$ in the range
 * [begin, begin + num_elems)
 *
 * \f[
 *     x \leftarrow \psi(x)
 * \f]
 *
 * where \f$\psi\f$ is cuda::kernel::psi_appr.
 *
 * @tparam Real A real type such as double or float.
 * @param range A DeviceMemory1D object holding the GPU memory to be updated
 * with psi_appr function.
 */
template <typename Real> void apply_psi(DeviceMemory1D<Real>& range);

/**
 * @brief Namespace containing update functions that are employed in
 * bnmf_algs::bld::bld_mult reimplemented in CUDA for higher efficiency.
 *
 * @todo Move this namespace to a new file (move all bld_mult functions to a
 * single file and divide bld_algs.cpp to several files?)
 */
namespace bld_mult {

/**
 * @brief Perform grad_plus update employed in bld_mult algorithm using CUDA.
 *
 * This function performs the grad_plus update employed in bld_mult algorithm
 * using CUDA.
 *
 * <B> All the given tensor and matrix objects are assumed to be in row-major
 * order </B>. This is the allocation order specified in defs.hpp file. All the
 * CUDA indexing operations are performed according to the row-major allocation
 * order.
 *
 * Row-major order means that the last index of a matrix/tensor objects changes
 * the fastest. This is easy to understand with matrices: Matrix is stored
 * in terms of its rows (0th row, then 1st row, ...). Allocation order of a 3D
 * tensor follows the same idea: Tensor is stored in terms of its fibers (depth
 * rows). Therefore, for a a 2x2x3 tensor is stored as follows (only writing the
 * indices):
 *
 * <blockquote>
 * (0,0,0) (0,0,1) (0,0,2) (0,1,0) (0,1,1) (0,1,2) (1,0,0) (1,0,1) (1,0,2)
 * (1,1,0) (1,1,1) (1,1,2)
 * </blockquote>
 *
 * @tparam Real Type of the entries of matrices and tensors such as double or
 * float.
 * @param S \f$S\f$ tensor on the GPU.
 * @param beta_eph beta_eph matrix on the GPU.
 * @param grad_plus grad_plus tensor on the GPU.
 */
template <typename Real>
void update_grad_plus(const DeviceMemory3D<Real>& S,
                      const DeviceMemory2D<Real>& beta_eph,
                      DeviceMemory3D<Real>& grad_plus);

template <typename Real>
void update_nom(const DeviceMemory2D<Real>& X_reciprocal,
                const DeviceMemory2D<Real>& grad_minus,
                const DeviceMemory3D<Real>& S, DeviceMemory2D<Real>& nom);

template <typename Real>
void update_denom(const DeviceMemory2D<Real>& X_reciprocal,
                  const DeviceMemory3D<Real>& grad_plus,
                  const DeviceMemory3D<Real>& S, DeviceMemory2D<Real>& denom);

template <typename Real>
void update_S(const DeviceMemory2D<Real>& X, const DeviceMemory2D<Real>& nom,
              const DeviceMemory2D<Real>& denom,
              const DeviceMemory2D<Real>& grad_minus,
              const DeviceMemory3D<Real>& grad_plus,
              const DeviceMemory2D<Real>& S_ijp, DeviceMemory3D<Real>& S);
} // namespace bld_mult
} // namespace cuda
} // namespace bnmf_algs
