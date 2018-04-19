#include "cuda/memory.hpp"
#include "cuda/tensor_ops.hpp"
#include "cuda/tensor_ops_kernels.hpp"
#include "cuda/util.hpp"
#include <array>
#include <cuda_runtime.h>

using namespace bnmf_algs;

template <typename Real> void cuda::apply_psi(DeviceMemory1D<Real>& range) {
    // grid/block dimensions
    const auto num_elems = range.dims()[0];
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = cuda::idiv_ceil(num_elems, threads_per_block);

    // apply kernel
    kernel::apply_psi<<<blocks_per_grid, threads_per_block>>>(range.data(),
                                                              num_elems);
    auto err = cudaGetLastError();
    BNMF_ASSERT(err == cudaSuccess, "Error running kernel in cuda::apply_psi");
}

template <typename T>
void cuda::tensor_sums(const cuda::DeviceMemory3D<T>& tensor,
                       std::array<cuda::DeviceMemory2D<T>, 3>& result_arr) {
    // input tensor properties
    const auto dims = tensor.dims();
    const size_t height = dims[0];
    const size_t width = dims[1];
    const size_t depth = dims[2];

    // block dimensions (number of threads per block axis)
    constexpr size_t block_size_height = 32;
    constexpr size_t block_size_width = 32;
    constexpr size_t block_size_depth = 1;
    dim3 block_dims(block_size_width, block_size_height, block_size_depth);

    constexpr size_t num_axes = 3;
    // launch each sum kernel on different CUDA stream
    std::array<cudaStream_t, num_axes> stream_arr;
    for (size_t i = 0; i < num_axes; ++i) {
        cudaStreamCreate(&stream_arr[i]);
    }

    // compute sum along each axis
    for (size_t axis = 0; axis < num_axes; ++axis) {
        // rows, cols, layers set when looked at the tensor from the face along
        // the current axis
        const size_t n_rows = (axis == 0) ? width : height;
        const size_t n_cols = (axis == 2) ? width : depth;
        const size_t n_layers = dims[axis];

        // kernel grid dimensions
        dim3 grid_dims(cuda::idiv_ceil(n_cols, block_size_width),
                       cuda::idiv_ceil(n_rows, block_size_height), 1);

        // launch asynchronous kernel
        kernel::sum_tensor3D<<<grid_dims, block_dims, 0, stream_arr[axis]>>>(
            tensor.pitched_ptr(), result_arr[axis].data(),
            result_arr[axis].pitch(), axis, n_rows, n_cols, n_layers);
    }

    // synchronize all streams
    for (size_t i = 0; i < num_axes; ++i) {
        cudaStreamSynchronize(stream_arr[i]);
        cudaStreamDestroy(stream_arr[i]);
    }
}

template <typename Real>
void cuda::bld_mult::update_grad_plus(
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
    kernel::update_grad_plus<<<grid_dims, block_dims>>>(
        S.pitched_ptr(), beta_eph.data(), beta_eph.pitch(),
        grad_plus.pitched_ptr(), y, x, z);
    auto err = cudaGetLastError();
    BNMF_ASSERT(err == cudaSuccess,
                "Error running kernel in cuda::bld_mult::update_grad_plus");
}

template <typename Real>
void cuda::bld_mult::update_nom(const DeviceMemory2D<Real>& X_reciprocal,
                                const DeviceMemory2D<Real>& grad_minus,
                                const DeviceMemory3D<Real>& S,
                                DeviceMemory2D<Real>& nom) {
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
    kernel::update_nom<<<grid_dims, block_dims>>>(
        S.pitched_ptr(), X_reciprocal.data(), X_reciprocal.pitch(),
        grad_minus.data(), grad_minus.pitch(), nom.data(), nom.pitch(), y, x,
        z);
    auto err = cudaGetLastError();
    BNMF_ASSERT(err == cudaSuccess,
                "Error running kernel in cuda::bld_mult::update_nom");
}

template <typename Real>
void cuda::bld_mult::update_denom(const DeviceMemory2D<Real>& X_reciprocal,
                                  const DeviceMemory3D<Real>& grad_plus,
                                  const DeviceMemory3D<Real>& S,
                                  DeviceMemory2D<Real>& denom) {
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
    kernel::update_denom<<<grid_dims, block_dims>>>(
        S.pitched_ptr(), X_reciprocal.data(), X_reciprocal.pitch(),
        grad_plus.pitched_ptr(), denom.data(), denom.pitch(), y, x, z);
    auto err = cudaGetLastError();
    BNMF_ASSERT(err == cudaSuccess,
                "Error running kernel in cuda::bld_mult::update_nom");
}

template <typename Real>
void cuda::bld_mult::update_S(const DeviceMemory2D<Real>& X,
                              const DeviceMemory2D<Real>& nom,
                              const DeviceMemory2D<Real>& denom,
                              const DeviceMemory2D<Real>& grad_minus,
                              const DeviceMemory3D<Real>& grad_plus,
                              const DeviceMemory2D<Real>& S_ijp,
                              DeviceMemory3D<Real>& S) {
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
    kernel::update_S<<<grid_dims, block_dims>>>(
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

// apply_psi
template void cuda::apply_psi(cuda::DeviceMemory1D<double>&);
template void cuda::apply_psi(cuda::DeviceMemory1D<float>&);

// tensor_sums
template void cuda::tensor_sums(const cuda::DeviceMemory3D<double>&,
                                std::array<cuda::DeviceMemory2D<double>, 3>&);
template void cuda::tensor_sums(const cuda::DeviceMemory3D<float>&,
                                std::array<cuda::DeviceMemory2D<float>, 3>&);
template void cuda::tensor_sums(const cuda::DeviceMemory3D<int>&,
                                std::array<cuda::DeviceMemory2D<int>, 3>&);
template void cuda::tensor_sums(const cuda::DeviceMemory3D<long>&,
                                std::array<cuda::DeviceMemory2D<long>, 3>&);
template void cuda::tensor_sums(const cuda::DeviceMemory3D<size_t>&,
                                std::array<cuda::DeviceMemory2D<size_t>, 3>&);

// update_grad_plus
template void
cuda::bld_mult::update_grad_plus(const cuda::DeviceMemory3D<double>&,
                                 const cuda::DeviceMemory2D<double>&,
                                 cuda::DeviceMemory3D<double>&);
template void
cuda::bld_mult::update_grad_plus(const cuda::DeviceMemory3D<float>&,
                                 const cuda::DeviceMemory2D<float>&,
                                 cuda::DeviceMemory3D<float>&);

// update_nom
template void
cuda::bld_mult::update_nom(const DeviceMemory2D<double>& X_reciprocal,
                           const DeviceMemory2D<double>& grad_minus,
                           const DeviceMemory3D<double>& S,
                           DeviceMemory2D<double>& nom);
template void
cuda::bld_mult::update_nom(const DeviceMemory2D<float>& X_reciprocal,
                           const DeviceMemory2D<float>& grad_minus,
                           const DeviceMemory3D<float>& S,
                           DeviceMemory2D<float>& nom);

// update_denom
template void
cuda::bld_mult::update_denom(const DeviceMemory2D<double>& X_reciprocal,
                             const DeviceMemory3D<double>& grad_plus,
                             const DeviceMemory3D<double>& S,
                             DeviceMemory2D<double>& denom);
template void
cuda::bld_mult::update_denom(const DeviceMemory2D<float>& X_reciprocal,
                             const DeviceMemory3D<float>& grad_plus,
                             const DeviceMemory3D<float>& S,
                             DeviceMemory2D<float>& denom);

// update_S
template void cuda::bld_mult::update_S(const DeviceMemory2D<double>& X,
                                       const DeviceMemory2D<double>& nom,
                                       const DeviceMemory2D<double>& denom,
                                       const DeviceMemory2D<double>& grad_minus,
                                       const DeviceMemory3D<double>& grad_plus,
                                       const DeviceMemory2D<double>& S_ijp,
                                       DeviceMemory3D<double>& S);
template void cuda::bld_mult::update_S(const DeviceMemory2D<float>& X,
                                       const DeviceMemory2D<float>& nom,
                                       const DeviceMemory2D<float>& denom,
                                       const DeviceMemory2D<float>& grad_minus,
                                       const DeviceMemory3D<float>& grad_plus,
                                       const DeviceMemory2D<float>& S_ijp,
                                       DeviceMemory3D<float>& S);
