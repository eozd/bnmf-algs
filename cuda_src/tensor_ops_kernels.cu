#include "cuda/tensor_ops_kernels.hpp"

using namespace bnmf_algs;


template <typename Scalar>
__global__ void cuda::kernel::sum_tensor3D(cudaPitchedPtr tensor, Scalar* out,
                                           size_t out_pitch, size_t axis,
                                           size_t n_rows, size_t n_cols,
                                           size_t n_layers) {
    // index for n_rows
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    // index for n_cols
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    // return if thread is not in the index
    if (not(i < n_rows && j < n_cols)) {
        return;
    }

    // offsets along each dimension
    size_t row_byte_offset = 0;   // offset to go to ith row
    size_t col_byte_offset = 0;   // offset to go to jth column
    size_t layer_byte_offset = 0; // offset to go to kth layer
    // offsets are assigned according to row-major 3D tensor layout
    if (axis == 0) {
        // sum over height
        row_byte_offset = tensor.pitch;
        col_byte_offset = sizeof(Scalar);
        layer_byte_offset = n_rows * tensor.pitch;
    } else if (axis == 1) {
        // sum over width
        row_byte_offset = n_layers * tensor.pitch;
        col_byte_offset = sizeof(Scalar);
        layer_byte_offset = tensor.pitch;
    } else if (axis == 2) {
        // sum over depth
        row_byte_offset = tensor.pitch * tensor.ysize;
        col_byte_offset = tensor.pitch;
        layer_byte_offset = sizeof(Scalar);
    }

    // find the address of the unique (i, j) entry in the current face
    size_t offset = row_byte_offset * i + col_byte_offset * j;
    const char* face_ij = ((char*)tensor.ptr + offset);

    // find the address of output matrix to sum the results
    offset = i * out_pitch;
    Scalar* out_ij = (Scalar*)((char*)out + offset) + j;

    // sum
    Scalar sum = Scalar();
    for (size_t layer = 0; layer < n_layers; ++layer) {
        const Scalar* face_ijk =
            (const Scalar*)(face_ij + layer * layer_byte_offset);
        sum += *face_ijk;
    }
    *out_ij = sum;
}
/************************ TEMPLATE INSTANTIATIONS *****************************/
// We need these because CUDA requires explicit instantiations of all template
// kernels.

// sum_tensor3D
template __global__ void
cuda::kernel::sum_tensor3D(cudaPitchedPtr tensor, double* out, size_t out_pitch,
                           size_t axis, size_t width, size_t height,
                           size_t depth);
template __global__ void
cuda::kernel::sum_tensor3D(cudaPitchedPtr tensor, float* out, size_t out_pitch,
                           size_t axis, size_t width, size_t height,
                           size_t depth);
template __global__ void cuda::kernel::sum_tensor3D(cudaPitchedPtr tensor,
                                                    int* out, size_t out_pitch,
                                                    size_t axis, size_t width,
                                                    size_t height,
                                                    size_t depth);
template __global__ void cuda::kernel::sum_tensor3D(cudaPitchedPtr tensor,
                                                    long* out, size_t out_pitch,
                                                    size_t axis, size_t width,
                                                    size_t height,
                                                    size_t depth);
template __global__ void
cuda::kernel::sum_tensor3D(cudaPitchedPtr tensor, size_t* out, size_t out_pitch,
                           size_t axis, size_t width, size_t height,
                           size_t depth);

