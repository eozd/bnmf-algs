#pragma once

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
} // namespace kernel
} // namespace cuda
} // namespace bnmf_algs
