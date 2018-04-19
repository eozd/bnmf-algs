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
 * @brief Sum the given 3D tensor along the given axis and write the results to
 * the corresponding index of the given 2D pitched matrix memory.
 *
 * This CUDA kernel computes the sum of a 3D tensor and writes the results to
 * the given 2D memory. Both the given tensor and the matrix object <b>must be
 * row-major</b>. Sum operation is performed by assigning a thread to each
 * element of the tensor face along the sum axis. For example, if we are summing
 * along the 2nd dimension, a thread is assigned to all the elements to the
 * matrix formed from the 0th and 1st dimension. Then, each thread computes the
 * sum in linear time and writes the result to their corresponding entry in the
 * output matrix.
 *
 * CUDA grid and block dimensions must be chosen so that each thread is
 * assigned the corresponding element of the 0th face of the tensor along the
 * sum axis. Additionally, n_rows, n_cols and n_layers parameters must be set
 * according to the sum axis. n_rows is the number of rows of the tensor face
 * . n_cols is the number of columns of the tensor face. n_layers is the depth
 * of the sum. In these definitions, tensor face can be visualized in 3D space
 * as the 2D plane that is orthogonal to the sum axis vector.
 *
 * @tparam Scalar Type of the tensor and matrix entries.
 * @param tensor CUDA pitched pointer pointing to a 3D pitched memory.
 * @param out Pointer to a 2D CUDA pitched matrix memory. Dimensions of the
 * output matrix must be chosen according to the sum axis.
 * @param out_pitch Pitch of the output 2D CUDA pitched matrix memory.
 * @param axis Sum axis. Must be 0, 1 or 2.
 * @param n_rows Number of rows of the tensor face.
 * @param n_cols Number of columns of the tensor face.
 * @param n_layers Depth of the sum, i.e. number of elements that must be summed
 * up to compute a single entry of the output matrix.
 */
template <typename Scalar>
__global__ void sum_tensor3D(cudaPitchedPtr tensor, Scalar* out,
                             size_t out_pitch, size_t axis, size_t n_rows,
                             size_t n_cols, size_t n_layers);

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

/**
 * @brief Perform nom_mult update employed in bld_mult algorithm using
 * tensors/matrices residing on a GPU device.
 *
 * This function applies nom_mult update to nom_mult matrix given as a
 * raw GPU pointer. <B> All the given
 * tensor and matrix pointers are assumed to store their objects in row-major
 * order. </B>
 *
 * @tparam Real Type of the entries of the matrices/tensors. Must be doueble or
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
 *
 * @tparam Real
 * @param S
 * @param X_reciprocal
 * @param X_reciprocal_pitch
 * @param grad_plus
 * @param denom_mult
 * @param denom_mult_pitch
 * @param width
 * @param height
 * @param depth
 */
template <typename Real>
__global__ void update_denom(cudaPitchedPtr S, const Real* X_reciprocal,
                             size_t X_reciprocal_pitch,
                             cudaPitchedPtr grad_plus, Real* denom_mult,
                             size_t denom_mult_pitch, size_t width,
                             size_t height, size_t depth);
} // namespace kernel
} // namespace cuda
} // namespace bnmf_algs
