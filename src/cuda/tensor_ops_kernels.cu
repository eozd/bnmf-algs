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
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (not(i < height && j < width & k < depth)) {
        return;
    }

    // S(i, j, :)
    size_t S_offset = S.pitch * S.ysize * i + j * S.pitch;
    const double* S_row = (double*)((char*)S.ptr + S_offset);

    // grad_plus(i, j, :)
    size_t grad_plus_offset =
        grad_plus.pitch * grad_plus.ysize * i + j * grad_plus.pitch;
    double* grad_plus_row = (double*)((char*)grad_plus.ptr + grad_plus_offset);

    // beta_eph(j, :)
    size_t beta_eph_offset = j * pitch;
    const double* beta_eph_row = (double*)((char*)beta_eph + beta_eph_offset);

    grad_plus_row[k] = psi_appr(beta_eph_row[k]) - psi_appr(S_row[k] + 1);
}

} // namespace kernel
} // namespace cuda
} // namespace bnmf_algs
