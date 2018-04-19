#include "cuda/tensor_ops_kernels.hpp"

using namespace bnmf_algs;

template <typename Real> __device__ Real cuda::kernel::psi_appr(Real x) {
    Real extra = 0;

    // write each case separately to minimize number of divisions as much as
    // possible
    if (x < 1) {
        const Real a = x + 1;
        const Real b = x + 2;
        const Real c = x + 3;
        const Real d = x + 4;
        const Real e = x + 5;
        const Real ab = a * b;
        const Real cd = c * d;
        const Real ex = e * x;

        extra = ((a + b) * cd * ex + ab * d * ex + ab * c * ex + ab * cd * x +
                 ab * cd * e) /
                (ab * cd * ex);
        x += 6;
    } else if (x < 2) {
        const Real a = x + 1;
        const Real b = x + 2;
        const Real c = x + 3;
        const Real d = x + 4;
        const Real ab = a * b;
        const Real cd = c * d;
        const Real dx = d * x;

        extra =
            ((a + b) * c * dx + ab * dx + ab * c * x + ab * cd) / (ab * cd * x);
        x += 5;
    } else if (x < 3) {
        const Real a = x + 1;
        const Real b = x + 2;
        const Real c = x + 3;
        const Real ab = a * b;
        const Real cx = c * x;

        extra = ((a + b) * cx + (c + x) * ab) / (ab * cx);
        x += 4;
    } else if (x < 4) {
        const Real a = x + 1;
        const Real b = x + 2;
        const Real ab = a * b;

        extra = ((a + b) * x + ab) / (ab * x);
        x += 3;
    } else if (x < 5) {
        const Real a = x + 1;

        extra = (a + x) / (a * x);
        x += 2;
    } else if (x < 6) {
        extra = 1 / x;
        x += 1;
    }

    Real x2 = x * x;
    Real x4 = x2 * x2;
    Real x6 = x4 * x2;
    Real x8 = x6 * x2;
    Real x10 = x8 * x2;
    Real x12 = x10 * x2;
    Real x13 = x12 * x;
    Real x14 = x13 * x;

    // write the result of the formula simplified using symbolic calculation
    // to minimize the number of divisions
    Real res =
        std::log(x) + (-360360 * x13 - 60060 * x12 + 6006 * x10 - 2860 * x8 +
                       3003 * x6 - 5460 * x4 + 15202 * x2 - 60060) /
                          (720720 * x14);

    return res - extra;
}

template <typename Real>
__global__ void cuda::kernel::apply_psi(Real* begin, size_t num_elems) {
    size_t blockId_grid = blockIdx.x + blockIdx.y * gridDim.x +
                          blockIdx.z * gridDim.x * gridDim.y;
    size_t threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    size_t id = blockId_grid * threads_per_block + threadIdx.x +
                threadIdx.y * blockDim.x +
                threadIdx.z * blockDim.x * blockDim.y;

    if (id < num_elems) {
        begin[id] = psi_appr(begin[id]);
    }
}

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

template <typename Real>
__global__ void
cuda::kernel::update_grad_plus(cudaPitchedPtr S, const Real* beta_eph,
                               size_t pitch, cudaPitchedPtr grad_plus,
                               size_t width, size_t height, size_t depth) {
    // index for rows
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    // index for columns
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    // index for depth
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (not(i < height && j < width & k < depth)) {
        return;
    }

    // S(i, j, :)
    size_t fiber_pitch = S.pitch;
    size_t num_cols = S.ysize;
    size_t roof_pitch = fiber_pitch * num_cols;
    size_t offset = roof_pitch * i + // go i roofs towards bottom
                    j * fiber_pitch; // go j fibers to the right
    const Real* S_fiber = (Real*)((char*)S.ptr + offset);

    // grad_plus(i, j, :) : S and grad_plus have exact same shapes
    Real* grad_plus_fiber = (Real*)((char*)grad_plus.ptr + offset);

    // beta_eph(j, :)
    offset = j * pitch; // go j rows towards bottom
    const Real* beta_eph_row = (Real*)((char*)beta_eph + offset);

    // grad_plus(i, j, k) = psi(beta_eph(j, k)) - psi(S(i, j, k) + 1);
    grad_plus_fiber[k] = psi_appr(beta_eph_row[k]) - psi_appr(S_fiber[k] + 1);
}

template <typename Real>
__global__ void cuda::kernel::update_nom(
    cudaPitchedPtr S, const Real* X_reciprocal, size_t X_reciprocal_pitch,
    const Real* grad_minus, size_t grad_minus_pitch, Real* nom_mult,
    size_t nom_mult_pitch, size_t width, size_t height, size_t depth) {

    // index for rows
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    // index for columns
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (not(i < height && j < width)) {
        return;
    }

    // S(i, j, :)
    size_t fiber_pitch = S.pitch;
    size_t num_cols = S.ysize;
    size_t roof_pitch = fiber_pitch * num_cols;
    size_t offset = roof_pitch * i + // go i roofs towards bottom
                    j * fiber_pitch; // go j fibers to the right
    const Real* S_ij = (Real*)((char*)S.ptr + offset);

    // X_reciprocal(i, j)
    offset = i * X_reciprocal_pitch;
    const Real* X_reciprocal_ij = (Real*)((char*)X_reciprocal + offset) + j;

    // grad_minus(i, :)
    offset = i * grad_minus_pitch;
    const Real* grad_minus_i = (Real*)((char*)grad_minus + offset);

    // nom_mult(i, j)
    offset = i * nom_mult_pitch;
    Real* nom_mult_ij = (Real*)((char*)nom_mult + offset) + j;

    Real sum = Real();
    for (size_t k = 0; k < depth; ++k) {
        sum += S_ij[k] * (*X_reciprocal_ij) * grad_minus_i[k];
    }
    *nom_mult_ij = sum;
}

template <typename Real>
__global__ void
cuda::kernel::update_denom(cudaPitchedPtr S, const Real* X_reciprocal,
                           size_t X_reciprocal_pitch, cudaPitchedPtr grad_plus,
                           Real* denom_mult, size_t denom_mult_pitch,
                           size_t width, size_t height, size_t depth) {

    // index for rows
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    // index for columns
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (not(i < height && j < width)) {
        return;
    }

    // S(i, j, :)
    size_t fiber_pitch = S.pitch;
    size_t num_cols = S.ysize;
    size_t roof_pitch = fiber_pitch * num_cols;
    size_t offset = roof_pitch * i + // go i roofs towards bottom
                    j * fiber_pitch; // go j fibers to the right
    const Real* S_ij = (Real*)((char*)S.ptr + offset);

    // grad_plus(i, j, :)
    const Real* grad_plus_ij = (Real*)((char*)grad_plus.ptr + offset);

    // X_reciprocal(i, j)
    offset = i * X_reciprocal_pitch;
    const Real* X_reciprocal_ij = (Real*)((char*)X_reciprocal + offset) + j;

    // denom_mult(i, j)
    offset = i * denom_mult_pitch;
    Real* denom_mult_ij = (Real*)((char*)denom_mult + offset) + j;

    Real sum = Real();
    for (size_t k = 0; k < depth; ++k) {
        sum += S_ij[k] * (*X_reciprocal_ij) * grad_plus_ij[k];
    }
    *denom_mult_ij = sum;
}

/************************ TEMPLATE INSTANTIATIONS *****************************/
// We need these because CUDA requires explicit instantiations of all template
// kernels.

// psi_appr
template __device__ double cuda::kernel::psi_appr(double);
template __device__ float cuda::kernel::psi_appr(float);

// apply_psi
template __global__ void cuda::kernel::apply_psi(double*, size_t);
template __global__ void cuda::kernel::apply_psi(float*, size_t);

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

// update_grad_plus
template __global__ void
cuda::kernel::update_grad_plus(cudaPitchedPtr S, const double* beta_eph,
                               size_t pitch, cudaPitchedPtr grad_plus,
                               size_t width, size_t height, size_t depth);
template __global__ void
cuda::kernel::update_grad_plus(cudaPitchedPtr S, const float* beta_eph,
                               size_t pitch, cudaPitchedPtr grad_plus,
                               size_t width, size_t height, size_t depth);

// update_nom
template __global__ void cuda::kernel::update_nom(
    cudaPitchedPtr S, const double* X_reciprocal, size_t X_reciprocal_pitch,
    const double* grad_minus, size_t grad_minus_pitch, double* nom_mult,
    size_t nom_mult_pitch, size_t width, size_t height, size_t depth);
template __global__ void cuda::kernel::update_nom(
    cudaPitchedPtr S, const float* X_reciprocal, size_t X_reciprocal_pitch,
    const float* grad_minus, size_t grad_minus_pitch, float* nom_mult,
    size_t nom_mult_pitch, size_t width, size_t height, size_t depth);

// update_denom
template __global__ void
cuda::kernel::update_denom(cudaPitchedPtr S, const double* X_reciprocal,
                           size_t X_reciprocal_pitch, cudaPitchedPtr grad_plus,
                           double* denom_mult, size_t denom_mult_pitch,
                           size_t width, size_t height, size_t depth);
template __global__ void
cuda::kernel::update_denom(cudaPitchedPtr S, const float* X_reciprocal,
                           size_t X_reciprocal_pitch, cudaPitchedPtr grad_plus,
                           float* denom_mult, size_t denom_mult_pitch,
                           size_t width, size_t height, size_t depth);
