#include "cuda/memory.hpp"
#include "cuda/tensor_ops.hpp"
#include "cuda/util.hpp"
#include "defs.hpp"

using namespace bnmf_algs;

template <typename T>
std::array<tensor_t<T, 2>, 3> cuda::tensor_sums(const tensor_t<T, 3>& tensor) {
    // input tensor properties
    const long x = tensor.dimension(0);
    const long y = tensor.dimension(1);
    const long z = tensor.dimension(2);
    const size_t in_num_elems = static_cast<size_t>(x * y * z);

    // since Eigen tensormap can't handle pitched memory, we need to use
    // contiguous 1D memory
    HostMemory1D<const T> host_tensor(tensor.data(), in_num_elems);
    DeviceMemory1D<T> device_tensor(in_num_elems);

    // copy input tensor to GPU
    copy1D(device_tensor, host_tensor, cudaMemcpyHostToDevice);

    // create GPU stream and device (default stream)
    Eigen::CudaStreamDevice stream;
    Eigen::GpuDevice dev(&stream);

    // sum axis
    shape<1> sum_axis;

    // map GPU tensor to Eigen (no copying)
    Eigen::TensorMap<tensor_t<T, 3>> in_tensor(device_tensor.data(), x, y, z);

    // results
    std::array<tensor_t<T, 2>, 3> result = {
        tensor_t<T, 2>(y, z), // along axis 0
        tensor_t<T, 2>(x, z), // along axis 1
        tensor_t<T, 2>(x, y)  // along axis 2
    };

    // memory maps to result tensors
    std::array<HostMemory1D<T>, 3> result_mems = {
        HostMemory1D<T>(result[0].data(), static_cast<size_t>(y * z)),
        HostMemory1D<T>(result[1].data(), static_cast<size_t>(x * z)),
        HostMemory1D<T>(result[2].data(), static_cast<size_t>(x * y))};

    // compute sum along each axis
    for (size_t axis = 0; axis < 3; ++axis) {
        long rows = (axis == 0) ? y : x;
        long cols = (axis == 2) ? y : z;
        const auto out_num_elems = static_cast<size_t>(rows * cols);

        // allocate memory on GPU for sum along current axis
        DeviceMemory1D<T> device_sum_tensor(out_num_elems);

        // map GPU tensor to Eigen (no copying)
        Eigen::TensorMap<tensor_t<T, 2>> out_tensor(device_sum_tensor.data(),
                                                    rows, cols);

        // axis to sum along
        sum_axis[0] = axis;

        // sum the tensor on GPU
        out_tensor.device(dev) = in_tensor.sum(sum_axis);
        cudaStreamSynchronize(stream.stream());

        // copy result back to main memory
        copy1D(result_mems[axis], device_sum_tensor, cudaMemcpyDeviceToHost);
    }

    return result;
}

template <typename Real> Real* cuda::apply_psi(Real* begin, size_t num_elems) {
    // host memory wrapper
    cuda::HostMemory1D<Real> host_data(begin, num_elems);

    // allocate device memory
    cuda::DeviceMemory1D<Real> device_data(num_elems);

    // copy range to GPU
    cuda::copy1D(device_data, host_data, cudaMemcpyHostToDevice);

    // grid/block dimensions
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = cuda::idiv_ceil(num_elems, threads_per_block);

    // apply kernel
    kernel::apply_psi<<<blocks_per_grid, threads_per_block>>>(
        device_data.data(), num_elems);
    BNMF_ASSERT(cudaGetLastError() == cudaSuccess,
                "Error running kernel in cuda::apply_psi");

    // copy from GPU to main memory
    cuda::copy1D(host_data, device_data, cudaMemcpyDeviceToHost);

    return begin;
}

template <typename T>
void cuda::bld_mult::update_grad_plus(const tensor_t<T, 3>& S,
                                      const matrix_t<T>& beta_eph,
                                      tensor_t<T, 3>& grad_plus) {
    // tensor dimensions
    const auto x = static_cast<size_t>(S.dimension(0));
    const auto y = static_cast<size_t>(S.dimension(1));
    const auto z = static_cast<size_t>(S.dimension(2));

    // create host memory wrappers
    HostMemory3D<const T> host_S(S.data(), x, y, z);
    HostMemory3D<T> host_grad_plus(grad_plus.data(), x, y, z);
    HostMemory2D<const T> host_beta_eph(beta_eph.data(), y, z);

    // allocate device memory
    DeviceMemory3D<T> device_S(x, y, z);
    DeviceMemory2D<T> device_beta_eph(y, z);
    DeviceMemory3D<T> device_grad_plus(x, y, z);

    // copy S to GPU
    copy3D(device_S, host_S, cudaMemcpyHostToDevice);

    // copy beta_eph to GPU
    copy2D(device_beta_eph, host_beta_eph, cudaMemcpyHostToDevice);

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
        device_S.pitched_ptr(), device_beta_eph.data(), device_beta_eph.pitch(),
        device_grad_plus.pitched_ptr(), y, x, z);
    BNMF_ASSERT(cudaGetLastError() == cudaSuccess,
                "Error running kernel in cuda::bld_mult::update_grad_plus");

    // copy result onto grad_plus
    copy3D(host_grad_plus, device_grad_plus, cudaMemcpyDeviceToHost);
}

/************************ TEMPLATE INSTANTIATIONS *****************************/
template std::array<tensor_t<double, 2>, 3>
cuda::tensor_sums(const tensor_t<double, 3>&);
template std::array<tensor_t<float, 2>, 3>
cuda::tensor_sums(const tensor_t<float, 3>&);
template std::array<tensor_t<int, 2>, 3>
cuda::tensor_sums(const tensor_t<int, 3>&);
template std::array<tensor_t<long, 2>, 3>
cuda::tensor_sums(const tensor_t<long, 3>&);
template std::array<tensor_t<size_t, 2>, 3>
cuda::tensor_sums(const tensor_t<size_t, 3>&);

template double* cuda::apply_psi(double*, size_t);
template float* cuda::apply_psi(float*, size_t);

template void cuda::bld_mult::update_grad_plus(const tensor_t<double, 3>&,
                                               const matrix_t<double>&,
                                               tensor_t<double, 3>&);
template void cuda::bld_mult::update_grad_plus(const tensor_t<float, 3>&,
                                               const matrix_t<float>&,
                                               tensor_t<float, 3>&);
