#pragma once
#include "bld_mult_cuda_funcs.hpp"

namespace bnmf_algs {
namespace details {
namespace bld_mult {
/**
 * @brief Namespace containing CUDA kernels used in bld_mult CUDA updates.
 */
namespace kernel {
/**
 * @brief Device function to return psi_appr of a real number.
 *
 * @tparam Real Type of the input parameter. Must be double or float.
 * @param x Value to apply psi_appr function.
 * @return psi_appr of x.
 *
 * @todo Check relocatable device code and move this function to a common
 * .cu file to be used by other files as well.
 */
template <typename Real> __device__ Real psi_appr(Real x);

/**
 * @brief Perform grad_plus update employed in bld_mult algorithm using
 * tensors/matrices residing on a GPU device.
 *
 * This function applies grad_plus update to grad_plus tensor given as a
 * cudaPitchedPtr using tensor S given as cudaPitchedPtr and beta_eph matrix
 * given as a device pointer and the pitch of the allocation. <B> All the given
 * tensor and matrix pointers are assumed to store their objects in row-major
 * order. </B> See bnmf_algs::cuda::bld_mult::update_grad_plus function
 * documentation for more information regarding row-major storage of
 * multidimensional tensors.
 *
 * @tparam Real Type of the entries of the matrices/tensors. Must be double or
 * float.
 * @param S Pitched pointer pointing to S tensor on the GPU device.
 * @param beta_eph Raw pointer pointing to beta_eph matrix on the GPU device.
 * @param pitch Pitch of the beta_eph matrix (number of bytes used to store a
 * single row of the matrix including padding bytes).
 * @param grad_plus Pitched pointer pointing to grad_plus tensor on the GPU
 * device.
 * @param width Width (1st dimension) of S and grad_plus tensors in terms of
 * elements.
 * @param height Height (0th dimension) of S and grad_plus tensors in terms of
 * elements.
 * @param depth Depth (2nd dimension) of S and grad_plus tensors in terms of
 * elements.
 */
template <typename Real>
__global__ void update_grad_plus(cudaPitchedPtr S, const Real* beta_eph,
                                 size_t pitch, cudaPitchedPtr grad_plus,
                                 size_t width, size_t height, size_t depth);

/**
 * @brief Perform nom_mult update employed in bld_mult algorithm using
 * tensors/matrices residing on a GPU device.
 *
 * This function applies nom_mult update to nom_mult matrix given as a
 * raw GPU pointer. <B> All the given
 * tensor and matrix pointers are assumed to store their objects in row-major
 * order. </B>
 *
 * @tparam Real Type of the entries of the matrices/tensors. Must be double or
 * float.
 * @param S Pitched pointer pointing to S tensor on the GPU device.
 * @param X_reciprocal Raw pointer pointing to X_reciprocal matrix on the GPU.
 * @param X_reciprocal_pitch Pitch of X_reciprocal matrix (number of bytes
 * used to store a single row of the matrix including padding bytes).
 * @param grad_minus Raw pointer pointing to grad_minus matrix on the GPU.
 * @param grad_minus_pitch Pitch of grad_minus matrix.
 * @param nom_mult Raw pointer pointing to nom_mult matrix on the GPU.
 * @param nom_mult_pitch Pitch of nom_mult matrix.
 * @param width Width (1st dimension) of S and grad_plus tensors in terms of
 * elements.
 * @param height Height (0th dimension) of S and grad_plus tensors in terms of
 * elements.
 * @param depth Depth (2nd dimension) of S and grad_plus tensors in terms of
 * elements.
 */
template <typename Real>
__global__ void update_nom(cudaPitchedPtr S, const Real* X_reciprocal,
                           size_t X_reciprocal_pitch, const Real* grad_minus,
                           size_t grad_minus_pitch, Real* nom_mult,
                           size_t nom_mult_pitch, size_t width, size_t height,
                           size_t depth);

/**
 * @brief Perform denom_mult update employed in bld_mult algorithm using
 * tensors/matrices residing on a GPU device.
 *
 * This function applies denom_mult update to denom_mult matrix given as a
 * raw GPU pointer. <B> All the given
 * tensor and matrix pointers are assumed to store their objects in row-major
 * order. </B>
 *
 * @tparam Real Type of the entries of the matrices/tensors. Must be double or
 * float.
 * @param S Pitched pointer pointing to S tensor on the GPU device.
 * @param X_reciprocal Raw pointer pointing to X_reciprocal matrix on the GPU.
 * @param X_reciprocal_pitch Pitch of X_reciprocal matrix (number of bytes
 * used to store a single row of the matrix including padding bytes).
 * @param grad_plus Pitched pointer pointing to grad_plus tensor on the GPU
 * device.
 * @param denom_mult Raw pointer pointing to denom_mult matrix on the GPU.
 * @param denom_mult_pitch Pitch of denom_mult matrix.
 * @param width Width (1st dimension) of S and grad_plus tensors in terms of
 * elements.
 * @param height Height (0th dimension) of S and grad_plus tensors in terms of
 * elements.
 * @param depth Depth (2nd dimension) of S and grad_plus tensors in terms of
 * elements.
 */
template <typename Real>
__global__ void update_denom(cudaPitchedPtr S, const Real* X_reciprocal,
                             size_t X_reciprocal_pitch,
                             cudaPitchedPtr grad_plus, Real* denom_mult,
                             size_t denom_mult_pitch, size_t width,
                             size_t height, size_t depth);

/**
 * @brief Perform S update employed in bld_mult algorithm using
 * tensors/matrices residing on a GPU device.
 *
 * This function applies S update to S tensor given as a cudaPitchedPtr. <B>
 * All the given tensor and matrix pointers are assumed to store their objects
 * in row-major order. </B>
 *
 * @tparam Real Type of the entries of the matrices/tensors. Must be double or
 * float.
 * @param X Raw pointer pointing to X matrix on the GPU.
 * @param X_pitch Pitch of X matrix (number of bytes used to store a single
 * row of the matrix including padding bytes).
 * @param nom_mult Raw pointer pointing to nom_mult matrix on the GPU.
 * @param nom_mult_pitch Pitch of nom_mult matrix
 * @param denom_mult Raw pointer pointing to denom_mult matrix on the GPU.
 * @param denom_mult_pitch Pitch of denom_mult matrix.
 * @param grad_minus Raw pointer pointing to grad_minus matrix on the GPU.
 * @param grad_minus_pitch Pitch of grad_minus matrix.
 * @param grad_plus Pitched pointer pointing to grad_plus tensor on the GPU.
 * @param S_ijp Raw pointer pointing to S_ijp matrix on the GPU.
 * @param S_ijp_pitch Pitch of S_ijp matrix.
 * @param S Pitched pointer pointing to S tensor on the GPU.
 * @param width Width (1st dimension) of S and grad_plus tensors in terms of
 * elements.
 * @param height Height (0th dimension) of S and grad_plus tensors in terms of
 * elements.
 * @param depth Depth (2nd dimension) of S and grad_plus tensors in terms of
 * elements.
 */
template <typename Real>
__global__ void
update_S(const Real* X, size_t X_pitch, const Real* nom_mult,
         size_t nom_mult_pitch, const Real* denom_mult, size_t denom_mult_pitch,
         const Real* grad_minus, size_t grad_minus_pitch,
         cudaPitchedPtr grad_plus, const Real* S_ijp, size_t S_ijp_pitch,
         cudaPitchedPtr S, size_t width, size_t height, size_t depth);
} // namespace kernel
} // namespace bld_mult
} // namespace details
} // namespace bnmf_algs
