#include "cuda/memory.hpp"
#include "cuda/tensor_ops.hpp"
#include "cuda/tensor_ops_kernels.hpp"
#include "cuda/util.hpp"
#include "defs.hpp"
#include <array>
#include <cuda_runtime.h>

using namespace bnmf_algs;

template <typename T>
void cuda::tensor_sums(const cuda::DeviceMemory1D<T>& tensor,
                       const shape<3>& dims,
                       std::array<cuda::DeviceMemory1D<T>, 3>& result_arr) {
    // input tensor properties
    const long x = dims[0];
    const long y = dims[1];
    const long z = dims[2];

    // create GPU stream and device (default stream)
    Eigen::CudaStreamDevice stream;
    Eigen::GpuDevice dev(&stream);

    // sum axis
    shape<1> sum_axis;

    // map GPU tensor to Eigen (no copying)
    Eigen::TensorMap<tensor_t<T, 3>> in_tensor(tensor.data(), x, y, z);

    // compute sum along each axis
    for (size_t axis = 0; axis < 3; ++axis) {
        long rows = (axis == 0) ? y : x;
        long cols = (axis == 2) ? y : z;

        // map GPU tensor to Eigen (no copying)
        Eigen::TensorMap<tensor_t<T, 2>> out_tensor(result_arr[axis].data(),
                                                    rows, cols);

        // axis to sum along
        sum_axis[0] = axis;

        // sum the tensor on GPU
        out_tensor.device(dev) = in_tensor.sum(sum_axis);
        cudaStreamSynchronize(stream.stream());
    }
}

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

/************************ TEMPLATE INSTANTIATIONS *****************************/
template void
cuda::tensor_sums<double>(const cuda::DeviceMemory1D<double>&, const shape<3>&,
                          std::array<cuda::DeviceMemory1D<double>, 3>&);
template void
cuda::tensor_sums<float>(const cuda::DeviceMemory1D<float>&, const shape<3>&,
                         std::array<cuda::DeviceMemory1D<float>, 3>&);
template void cuda::tensor_sums<int>(const cuda::DeviceMemory1D<int>&,
                                     const shape<3>&,
                                     std::array<cuda::DeviceMemory1D<int>, 3>&);
template void
cuda::tensor_sums<long>(const cuda::DeviceMemory1D<long>&, const shape<3>&,
                        std::array<cuda::DeviceMemory1D<long>, 3>&);
template void
cuda::tensor_sums<size_t>(const cuda::DeviceMemory1D<size_t>&, const shape<3>&,
                          std::array<cuda::DeviceMemory1D<size_t>, 3>&);

template void cuda::apply_psi<double>(cuda::DeviceMemory1D<double>&);
template void cuda::apply_psi<float>(cuda::DeviceMemory1D<float>&);

template void
cuda::bld_mult::update_grad_plus<double>(const cuda::DeviceMemory3D<double>&,
                                         const cuda::DeviceMemory2D<double>&,
                                         cuda::DeviceMemory3D<double>&);

template void
cuda::bld_mult::update_grad_plus<float>(const cuda::DeviceMemory3D<float>&,
                                        const cuda::DeviceMemory2D<float>&,
                                        cuda::DeviceMemory3D<float>&);
