#pragma once

#include "cuda/memory.hpp"

namespace bnmf_algs {
namespace details {
namespace bld_mult {
/**
 * @brief Perform grad_plus update employed in bld_mult algorithm using CUDA.
 *
 * This function performs the grad_plus update employed in bld_mult algorithm
 * using CUDA. To learn more about grad_plus update, refer to
 * details::bld_mult::update_grad_plus function documentation.
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
void update_grad_plus_cuda(const cuda::DeviceMemory3D<Real>& S,
                           const cuda::DeviceMemory2D<Real>& beta_eph,
                           cuda::DeviceMemory3D<Real>& grad_plus);

/**
 * @brief Perform nom_mult update employed in bld_mult algorithm using CUDA.
 *
 * This function performs the nom_mult update employed in bld_mult algorithm
 * using CUDA. To learn more about nom_mult update, refer to
 * details::bld_mult::update_nom_mult function documentation.
 *
 * <B> All the given tensor and matrix objects are assumed to be in row-major
 * order </B>. This is the allocation order specified in defs.hpp file. All the
 * CUDA indexing operations are performed according to the row-major allocation
 * order.
 *
 * @tparam Real Type of the entries of matrices and tensors such as double or
 * float.
 * @param X_reciprocal Matrix \f$\hat{X}\f$ where \f$\hat{X}_{ij} =
 * \frac{1}{X_{ij}}\f$. Should reside on the GPU.
 * @param grad_minus grad_minus matrix used in bld_mult on the GPU.
 * @param S \f$S\f$ tensor on the GPU.
 * @param nom nom_mult matrix on the GPU.
 */
template <typename Real>
void update_nom_cuda(const cuda::DeviceMemory2D<Real>& X_reciprocal,
                     const cuda::DeviceMemory2D<Real>& grad_minus,
                     const cuda::DeviceMemory3D<Real>& S,
                     cuda::DeviceMemory2D<Real>& nom);

/**
 * @brief Perform denom update employed in bld_mult algorithm using CUDA.
 *
 * This function performs the denom update employed in bld_mult algorithm
 * using CUDA. To learn more about denom update, refer to
 * details::bld_mult::update_denom_mult function documentation.
 *
 * <B> All the given tensor and matrix objects are assumed to be in row-major
 * order </B>. This is the allocation order specified in defs.hpp file. All the
 * CUDA indexing operations are performed according to the row-major allocation
 * order.
 *
 * @tparam Real Type of the entries of matrices and tensors such as double or
 * float.
 * @param X_reciprocal Matrix \f$\hat{X}\f$ where \f$\hat{X}_{ij} =
 * \frac{1}{X_{ij}}\f$. Should reside on the GPU.
 * @param grad_plus grad_plus tensor used in bld_mult on the GPU.
 * @param S \f$S\f$ tensor on the GPU.
 * @param denom denom_mult matrix on the GPU.
 */
template <typename Real>
void update_denom_cuda(const cuda::DeviceMemory2D<Real>& X_reciprocal,
                       const cuda::DeviceMemory3D<Real>& grad_plus,
                       const cuda::DeviceMemory3D<Real>& S,
                       cuda::DeviceMemory2D<Real>& denom);

/**
 * @brief Perform S update employed in bld_mult algorithm using CUDA.
 *
 * This function performs the S update employed in bld_mult algorithm
 * using CUDA. To learn more about S update, refer to
 * details::bld_mult::update_S function documentation.
 *
 * <B> All the given tensor and matrix objects are assumed to be in row-major
 * order </B>. This is the allocation order specified in defs.hpp file. All the
 * CUDA indexing operations are performed according to the row-major allocation
 * order.
 *
 * @tparam Real Type of the entries of matrices and tensors such as double or
 * float.
 * @param X Input X matrix on the GPU.
 * @param nom nom_mult.
 * @param denom denom_mult.
 * @param grad_minus grad_minus matrix.
 * @param grad_plus grad_plus tensor used in bld_mult on the GPU.
 * @param S_ijp Matrix formed by summing S tensor along its axis 2. (last axis)
 * @param S \f$S\f$ tensor on the GPU.
 */
template <typename Real>
void update_S_cuda(const cuda::DeviceMemory2D<Real>& X,
                   const cuda::DeviceMemory2D<Real>& nom,
                   const cuda::DeviceMemory2D<Real>& denom,
                   const cuda::DeviceMemory2D<Real>& grad_minus,
                   const cuda::DeviceMemory3D<Real>& grad_plus,
                   const cuda::DeviceMemory2D<Real>& S_ijp,
                   cuda::DeviceMemory3D<Real>& S);

} // namespace bld_mult
} // namespace details
} // namespace bnmf_algs
