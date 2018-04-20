#include "bld/bld_mult/bld_mult_cuda_funcs.hpp"
#include "bld/bld_mult/bld_mult_cuda_kernels.hpp"
#include "cuda/util.hpp"

using namespace bnmf_algs;

template <typename Real>
void details::bld_mult_update_grad_plus_cuda(
    const cuda::DeviceMemory3D<Real>& S,
    const cuda::DeviceMemory2D<Real>& beta_eph,
    cuda::DeviceMemory3D<Real>& grad_plus) {
    // tensor dimensions
    const auto x = S.dims()[0];
    const auto y = S.dims()[1];
    const auto z = S.dims()[2];

    // block dimensions (number of threads per block axis)
    constexpr size_t block_size_x = 16;
    constexpr size_t block_size_y = 16;
    constexpr size_t block_size_z = 4;
    dim3 block_dims(block_size_y, block_size_x, block_size_z);
    dim3 grid_dims(cuda::idiv_ceil(y, block_size_y),
                   cuda::idiv_ceil(x, block_size_x),
                   cuda::idiv_ceil(z, block_size_z));

    // run kernel
    kernel::bld_mult_update_grad_plus<<<grid_dims, block_dims>>>(
        S.pitched_ptr(), beta_eph.data(), beta_eph.pitch(),
        grad_plus.pitched_ptr(), y, x, z);
    auto err = cudaGetLastError();
    BNMF_ASSERT(err == cudaSuccess,
                "Error running kernel in cuda::bld_mult::update_grad_plus");
}

template <typename Real>
void details::bld_mult_update_nom_cuda(
    const cuda::DeviceMemory2D<Real>& X_reciprocal,
    const cuda::DeviceMemory2D<Real>& grad_minus,
    const cuda::DeviceMemory3D<Real>& S, cuda::DeviceMemory2D<Real>& nom) {
    // tensor dimensions
    const auto x = S.dims()[0];
    const auto y = S.dims()[1];
    const auto z = S.dims()[2];

    // block dimensions (number of threads per block axis)
    constexpr size_t block_size_x = 32;
    constexpr size_t block_size_y = 32;
    constexpr size_t block_size_z = 1;
    dim3 block_dims(block_size_y, block_size_x, block_size_z);
    dim3 grid_dims(cuda::idiv_ceil(y, block_size_y),
                   cuda::idiv_ceil(x, block_size_x), 1);

    // run kernel
    kernel::bld_mult_update_nom<<<grid_dims, block_dims>>>(
        S.pitched_ptr(), X_reciprocal.data(), X_reciprocal.pitch(),
        grad_minus.data(), grad_minus.pitch(), nom.data(), nom.pitch(), y, x,
        z);
    auto err = cudaGetLastError();
    BNMF_ASSERT(err == cudaSuccess,
                "Error running kernel in cuda::bld_mult::update_nom");
}

template <typename Real>
void details::bld_mult_update_denom_cuda(
    const cuda::DeviceMemory2D<Real>& X_reciprocal,
    const cuda::DeviceMemory3D<Real>& grad_plus,
    const cuda::DeviceMemory3D<Real>& S, cuda::DeviceMemory2D<Real>& denom) {
    // tensor dimensions
    const auto x = S.dims()[0];
    const auto y = S.dims()[1];
    const auto z = S.dims()[2];

    // block dimensions (number of threads per block axis)
    constexpr size_t block_size_x = 32;
    constexpr size_t block_size_y = 32;
    constexpr size_t block_size_z = 1;
    dim3 block_dims(block_size_y, block_size_x, block_size_z);
    dim3 grid_dims(cuda::idiv_ceil(y, block_size_y),
                   cuda::idiv_ceil(x, block_size_x), 1);

    // run kernel
    kernel::bld_mult_update_denom<<<grid_dims, block_dims>>>(
        S.pitched_ptr(), X_reciprocal.data(), X_reciprocal.pitch(),
        grad_plus.pitched_ptr(), denom.data(), denom.pitch(), y, x, z);
    auto err = cudaGetLastError();
    BNMF_ASSERT(err == cudaSuccess,
                "Error running kernel in cuda::bld_mult::update_nom");
}

template <typename Real>
void details::bld_mult_update_S_cuda(
    const cuda::DeviceMemory2D<Real>& X, const cuda::DeviceMemory2D<Real>& nom,
    const cuda::DeviceMemory2D<Real>& denom,
    const cuda::DeviceMemory2D<Real>& grad_minus,
    const cuda::DeviceMemory3D<Real>& grad_plus,
    const cuda::DeviceMemory2D<Real>& S_ijp, cuda::DeviceMemory3D<Real>& S) {
    // tensor dimensions
    const auto x = S.dims()[0];
    const auto y = S.dims()[1];
    const auto z = S.dims()[2];

    // block dimensions (number of threads per block axis)
    constexpr size_t block_size_x = 16;
    constexpr size_t block_size_y = 16;
    constexpr size_t block_size_z = 4;
    dim3 block_dims(block_size_y, block_size_x, block_size_z);
    dim3 grid_dims(cuda::idiv_ceil(y, block_size_y),
                   cuda::idiv_ceil(x, block_size_x),
                   cuda::idiv_ceil(z, block_size_z));

    // run kernel
    kernel::bld_mult_update_S<<<grid_dims, block_dims>>>(
        X.data(), X.pitch(), nom.data(), nom.pitch(), denom.data(),
        denom.pitch(), grad_minus.data(), grad_minus.pitch(),
        grad_plus.pitched_ptr(), S_ijp.data(), S_ijp.pitch(), S.pitched_ptr(),
        y, x, z);
    auto err = cudaGetLastError();
    BNMF_ASSERT(err == cudaSuccess,
                "Error running kernel in cuda::bld_mult::update_grad_plus");
}

/************************ TEMPLATE INSTANTIATIONS *****************************/
// We need these because nvcc requires explicit instantiations of all template
// functions.

// update_grad_plus
template void
details::bld_mult_update_grad_plus_cuda(const cuda::DeviceMemory3D<double>&,
                                        const cuda::DeviceMemory2D<double>&,
                                        cuda::DeviceMemory3D<double>&);
template void
details::bld_mult_update_grad_plus_cuda(const cuda::DeviceMemory3D<float>&,
                                        const cuda::DeviceMemory2D<float>&,
                                        cuda::DeviceMemory3D<float>&);

// update_nom
template void details::bld_mult_update_nom_cuda(
    const cuda::DeviceMemory2D<double>& X_reciprocal,
    const cuda::DeviceMemory2D<double>& grad_minus,
    const cuda::DeviceMemory3D<double>& S, cuda::DeviceMemory2D<double>& nom);
template void details::bld_mult_update_nom_cuda(
    const cuda::DeviceMemory2D<float>& X_reciprocal,
    const cuda::DeviceMemory2D<float>& grad_minus,
    const cuda::DeviceMemory3D<float>& S, cuda::DeviceMemory2D<float>& nom);

// update_denom
template void details::bld_mult_update_denom_cuda(
    const cuda::DeviceMemory2D<double>& X_reciprocal,
    const cuda::DeviceMemory3D<double>& grad_plus,
    const cuda::DeviceMemory3D<double>& S, cuda::DeviceMemory2D<double>& denom);
template void details::bld_mult_update_denom_cuda(
    const cuda::DeviceMemory2D<float>& X_reciprocal,
    const cuda::DeviceMemory3D<float>& grad_plus,
    const cuda::DeviceMemory3D<float>& S, cuda::DeviceMemory2D<float>& denom);

// update_S
template void
details::bld_mult_update_S_cuda(const cuda::DeviceMemory2D<double>& X,
                                const cuda::DeviceMemory2D<double>& nom,
                                const cuda::DeviceMemory2D<double>& denom,
                                const cuda::DeviceMemory2D<double>& grad_minus,
                                const cuda::DeviceMemory3D<double>& grad_plus,
                                const cuda::DeviceMemory2D<double>& S_ijp,
                                cuda::DeviceMemory3D<double>& S);
template void
details::bld_mult_update_S_cuda(const cuda::DeviceMemory2D<float>& X,
                                const cuda::DeviceMemory2D<float>& nom,
                                const cuda::DeviceMemory2D<float>& denom,
                                const cuda::DeviceMemory2D<float>& grad_minus,
                                const cuda::DeviceMemory3D<float>& grad_plus,
                                const cuda::DeviceMemory2D<float>& S_ijp,
                                cuda::DeviceMemory3D<float>& S);
