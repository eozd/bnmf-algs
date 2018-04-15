#include "cuda/tensor_ops_kernels.hpp"
#include <cmath>
#include <device_launch_parameters.h>

namespace bnmf_algs {
namespace cuda {
namespace kernel {

__device__ double psi_appr(double x) {
    double extra = 0;

    // write each case separately to minimize number of divisions as much as
    // possible
    if (x < 1) {
        const double a = x + 1;
        const double b = x + 2;
        const double c = x + 3;
        const double d = x + 4;
        const double e = x + 5;
        const double ab = a * b;
        const double cd = c * d;
        const double ex = e * x;

        extra = ((a + b) * cd * ex + ab * d * ex + ab * c * ex + ab * cd * x +
                 ab * cd * e) /
                (ab * cd * ex);
        x += 6;
    } else if (x < 2) {
        const double a = x + 1;
        const double b = x + 2;
        const double c = x + 3;
        const double d = x + 4;
        const double ab = a * b;
        const double cd = c * d;
        const double dx = d * x;

        extra =
            ((a + b) * c * dx + ab * dx + ab * c * x + ab * cd) / (ab * cd * x);
        x += 5;
    } else if (x < 3) {
        const double a = x + 1;
        const double b = x + 2;
        const double c = x + 3;
        const double ab = a * b;
        const double cx = c * x;

        extra = ((a + b) * cx + (c + x) * ab) / (ab * cx);
        x += 4;
    } else if (x < 4) {
        const double a = x + 1;
        const double b = x + 2;
        const double ab = a * b;

        extra = ((a + b) * x + ab) / (ab * x);
        x += 3;
    } else if (x < 5) {
        const double a = x + 1;

        extra = (a + x) / (a * x);
        x += 2;
    } else if (x < 6) {
        extra = 1 / x;
        x += 1;
    }

    const double x2 = x * x;
    const double x4 = x2 * x2;
    const double x6 = x4 * x2;
    const double x8 = x6 * x2;
    const double x10 = x8 * x2;
    const double x12 = x10 * x2;
    const double x13 = x12 * x;
    const double x14 = x13 * x;

    // write the result of the formula simplified using symbolic calculation
    // to minimize the number of divisions
    const double res =
        std::log(x) + (-360360 * x13 - 60060 * x12 + 6006 * x10 - 2860 * x8 +
                       3003 * x6 - 5460 * x4 + 15202 * x2 - 60060) /
                          (720720 * x14);

    return res - extra;
}

__global__ void apply_psi(double* begin, size_t num_elems) {
    size_t i = blockDim.x * blockIdx.y + threadIdx.x;
    if (i < num_elems) {
        begin[i] = psi_appr(begin[i]);
    }
}

__global__ void update_grad_plus(const cudaPitchedPtr S, const double* beta_eph,
                                 const size_t pitch, cudaPitchedPtr grad_plus,
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
    const double* S_fiber = (double*)((char*)S.ptr + offset);

    // grad_plus(i, j, :) : S and grad_plus have exact same shapes
    double* grad_plus_fiber = (double*)((char*)grad_plus.ptr + offset);

    // beta_eph(j, :)
    offset = j * pitch; // go j rows towards bottom
    const double* beta_eph_row = (double*)((char*)beta_eph + offset);

    // grad_plus(i, j, k) = psi(beta_eph(j, k)) - psi(S(i, j, k) + 1);
    grad_plus_fiber[k] = psi_appr(beta_eph_row[k]) - psi_appr(S_fiber[k] + 1);
}

} // namespace kernel
} // namespace cuda
} // namespace bnmf_algs
