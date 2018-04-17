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

/************************ TEMPLATE INSTANTIATIONS *****************************/
template __device__ double cuda::kernel::psi_appr(double);
template __device__ float cuda::kernel::psi_appr(float);

template __global__ void cuda::kernel::apply_psi(double*, size_t);
template __global__ void cuda::kernel::apply_psi(float*, size_t);

template __global__ void
cuda::kernel::update_grad_plus(cudaPitchedPtr S, const double* beta_eph,
                               size_t pitch, cudaPitchedPtr grad_plus,
                               size_t width, size_t height, size_t depth);
template __global__ void
cuda::kernel::update_grad_plus(cudaPitchedPtr S, const float* beta_eph,
                               size_t pitch, cudaPitchedPtr grad_plus,
                               size_t width, size_t height, size_t depth);
