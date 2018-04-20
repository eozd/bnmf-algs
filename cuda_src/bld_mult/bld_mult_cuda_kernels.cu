#include "bld/bld_mult/bld_mult_cuda_kernels.hpp"
#include "cuda/tensor_ops_kernels.hpp"

#include <device_launch_parameters.h>

using namespace bnmf_algs;

template <typename Real>
__device__ Real details::bld_mult::kernel::psi_appr(Real x) {
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
__global__ void details::bld_mult::kernel::update_grad_plus(
    cudaPitchedPtr S, const Real* beta_eph, size_t pitch,
    cudaPitchedPtr grad_plus, size_t width, size_t height, size_t depth) {
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
__global__ void details::bld_mult::kernel::update_nom(
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
__global__ void details::bld_mult::kernel::update_denom(
    cudaPitchedPtr S, const Real* X_reciprocal, size_t X_reciprocal_pitch,
    cudaPitchedPtr grad_plus, Real* denom_mult, size_t denom_mult_pitch,
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

template <typename Real>
__global__ void details::bld_mult::kernel::update_S(
    const Real* X, size_t X_pitch, const Real* nom_mult, size_t nom_mult_pitch,
    const Real* denom_mult, size_t denom_mult_pitch, const Real* grad_minus,
    size_t grad_minus_pitch, cudaPitchedPtr grad_plus, const Real* S_ijp,
    size_t S_ijp_pitch, cudaPitchedPtr S, size_t width, size_t height,
    size_t depth) {
    constexpr double eps = 1e-50;
    // index for rows
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    // index for columns
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    // index for depth
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (not(i < height && j < width & k < depth)) {
        return;
    }

    // S(i, j, k)
    size_t fiber_pitch = S.pitch;
    size_t num_cols = S.ysize;
    size_t roof_pitch = fiber_pitch * num_cols;
    size_t offset = roof_pitch * i + // go i roofs towards bottom
                    j * fiber_pitch; // go j fibers to the right
    Real* S_ijk = (Real*)((char*)S.ptr + offset) + k;

    // grad_plus(i, j, k)
    Real grad_plus_ijk = *((Real*)((char*)grad_plus.ptr + offset) + k);

    // grad_minus(i, k)
    offset = i * grad_minus_pitch;
    Real grad_minus_ik = *((Real*)((char*)grad_minus + offset) + k);

    // X(i, j)
    offset = i * X_pitch;
    Real X_ij = *((Real*)((char*)X + offset) + j);

    // nom_mult(i, j)
    offset = i * nom_mult_pitch;
    Real nom_mult_ij = *((Real*)((char*)nom_mult + offset) + j);

    // denom_mult(i, j)
    offset = i * denom_mult_pitch;
    Real denom_mult_ij = *((Real*)((char*)denom_mult + offset) + j);

    // S_ijp(i, j)
    offset = i * S_ijp_pitch;
    Real S_ijp_ij = *((Real*)((char*)S_ijp + offset) + j);

    // calculation
    *S_ijk *= (grad_plus_ijk + nom_mult_ij) * (X_ij / (S_ijp_ij + eps)) /
              (grad_minus_ik + denom_mult_ij + eps);
}

/************************ TEMPLATE INSTANTIATIONS *****************************/
// We need these because CUDA requires explicit instantiations of all template
// kernels.
// psi_appr
template __device__ double details::bld_mult::kernel::psi_appr(double);
template __device__ float details::bld_mult::kernel::psi_appr(float);

// update_grad_plus
template __global__ void details::bld_mult::kernel::update_grad_plus(
    cudaPitchedPtr S, const double* beta_eph, size_t pitch,
    cudaPitchedPtr grad_plus, size_t width, size_t height, size_t depth);
template __global__ void details::bld_mult::kernel::update_grad_plus(
    cudaPitchedPtr S, const float* beta_eph, size_t pitch,
    cudaPitchedPtr grad_plus, size_t width, size_t height, size_t depth);

// update_nom
template __global__ void details::bld_mult::kernel::update_nom(
    cudaPitchedPtr S, const double* X_reciprocal, size_t X_reciprocal_pitch,
    const double* grad_minus, size_t grad_minus_pitch, double* nom_mult,
    size_t nom_mult_pitch, size_t width, size_t height, size_t depth);
template __global__ void details::bld_mult::kernel::update_nom(
    cudaPitchedPtr S, const float* X_reciprocal, size_t X_reciprocal_pitch,
    const float* grad_minus, size_t grad_minus_pitch, float* nom_mult,
    size_t nom_mult_pitch, size_t width, size_t height, size_t depth);

// update_denom
template __global__ void details::bld_mult::kernel::update_denom(
    cudaPitchedPtr S, const double* X_reciprocal, size_t X_reciprocal_pitch,
    cudaPitchedPtr grad_plus, double* denom_mult, size_t denom_mult_pitch,
    size_t width, size_t height, size_t depth);
template __global__ void details::bld_mult::kernel::update_denom(
    cudaPitchedPtr S, const float* X_reciprocal, size_t X_reciprocal_pitch,
    cudaPitchedPtr grad_plus, float* denom_mult, size_t denom_mult_pitch,
    size_t width, size_t height, size_t depth);

// update_S
template __global__ void details::bld_mult::kernel::update_S(
    const double* X, size_t X_pitch, const double* nom_mult,
    size_t nom_mult_pitch, const double* denom_mult, size_t denom_mult_pitch,
    const double* grad_minus, size_t grad_minus_pitch, cudaPitchedPtr grad_plus,
    const double* S_ijp, size_t S_ijp_pitch, cudaPitchedPtr S, size_t width,
    size_t height, size_t depth);
template __global__ void details::bld_mult::kernel::update_S(
    const float* X, size_t X_pitch, const float* nom_mult,
    size_t nom_mult_pitch, const float* denom_mult, size_t denom_mult_pitch,
    const float* grad_minus, size_t grad_minus_pitch, cudaPitchedPtr grad_plus,
    const float* S_ijp, size_t S_ijp_pitch, cudaPitchedPtr S, size_t width,
    size_t height, size_t depth);
