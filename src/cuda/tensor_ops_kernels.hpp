#include "defs.hpp"
#include <cuda.h>
#include <device_launch_parameters.h>

namespace bnmf_algs {
namespace cuda {
/**
 * @brief Namespace containing CUDA kernel and device functions.
 */
namespace kernel {
/**
 * @brief Device function to return psi_appr of a real number.
 *
 * @tparam Real Type of the input parameter. Must be double or float.
 * @param x Value to apply psi_appr function.
 * @return psi_appr of x.
 */
template <typename Real> __device__ Real psi_appr(Real x);

/**
 * @brief Device kernel to apply psi_appr function to all the values in the
 * given range [begin, begin + num_elems) using CUDA GPU grids and blocks.
 *
 * This function applies psi_appr function to all the values in the range in
 * parallel. Index of a given thread is calculated as
 *
 * @code
 *     size_t i = blockDim.x * blockIdx.y + threadIdx.x;
 * @endcode
 *
 * @tparam Real Type of the input parameter. Must be double or float.
 * @param begin Device pointer indicating the beginning of the range residing
 * in a GPU device.
 * @param num_elems Number of elements in the GPU device to apply psi_appr func.
 */
template <typename Real>
__global__ void apply_psi(Real* begin, size_t num_elems);

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
 * @tparam Real Type of the entries of the matrices/tensors. Must be doueble or
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

} // namespace kernel
} // namespace cuda
} // namespace bnmf_algs
