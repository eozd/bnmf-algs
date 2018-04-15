#pragma once

#include "defs.hpp"
#include <array>

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

/**
 * @brief Apply cuda::kernel::psi_appr function to every element in the range
 * [begin, begin + num_elems) in-place using CUDA.
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
 * apply_psi executes by copying the range to the GPU, executing psi application
 * procedure completely in parallel and copying the results in the GPU onto the
 * range beginning in the pointer parameter begin.
 *
 * @param begin Pointer to the beginning of the range to apply util::psi_appr
 * function.
 * @param num_elems Number of elements in the range [begin, begin + num_elems).
 *
 * @return Pointer to the beginning of the altered range. This pointer is
 * equal to the given pointer parameter begin.
 */
double* apply_psi(double* begin, size_t num_elems);

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
 * by copying S tensor and beta_eph matrix to GPU and performing the update in
 * parallel using CUDA. Afterwards, the result of the update is copied back onto
 * the given tensor parameter grad_plus.
 *
 * <B> All the given tensor and matrix objects are assumed to be in row-major
 * order </B>. This is the allocation order specified in defs.hpp file. All the
 * CUDA memory allocations and indexing operations are performed according to
 * the row-major allocation order.
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
 * @param S \f$S\f$ tensor, that is the output of bld_mult algorithm.
 * @param beta_eph beta_eph matrix used during bld_mult algorithm.
 * @param grad_plus grad_plus tensor that will store the results of grad_plus
 * update.
 */
void update_grad_plus(const tensord<3>& S, const matrixd& beta_eph,
                      tensord<3>& grad_plus);

} // namespace bld_mult
} // namespace cuda
} // namespace bnmf_algs
