#include "cuda/memory.hpp"
#include "cuda/tensor_ops.hpp"
#include "cuda/tensor_ops_kernels.hpp"
#include "defs.hpp"
#include <cassert>
#include <cuda_runtime.h>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace bnmf_algs;

void cuda::init() {
    cudaError_t err = cudaSuccess;
    err = cudaSetDevice(0);
    assert(err == cudaSuccess);
}

std::array<tensord<2>, 3> cuda::tensor_sums(const tensord<3>& tensor) {

    // input tensor properties
    const long x = tensor.dimension(0);
    const long y = tensor.dimension(1);
    const long z = tensor.dimension(2);
    const auto in_num_elems = static_cast<size_t>(x * y * z);
    const size_t in_size = in_num_elems * sizeof(double);

    cudaError_t err = cudaSuccess;

    // allocate memory for input tensor on GPU
    double* in_data;
    err = cudaMalloc((void**)(&in_data), in_size);
    assert(err == cudaSuccess);

    // copy input tensor to GPU
    err = cudaMemcpy(in_data, tensor.data(), in_size, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);

    // create GPU stream and device (default stream)
    Eigen::CudaStreamDevice stream;
    Eigen::GpuDevice dev(&stream);

    // sum axis
    shape<1> sum_axis;

    // map GPU tensor to Eigen (no copying)
    Eigen::TensorMap<tensord<3>> in_tensor(in_data, x, y, z);

    // results
    std::array<tensord<2>, 3> result = {
        tensord<2>(y, z), // along axis 0
        tensord<2>(x, z), // along axis 1
        tensord<2>(x, y)  // along axis 2
    };

    // compute sum along each axis
    for (size_t axis = 0; axis < 3; ++axis) {
        long rows = (axis == 0) ? y : x;
        long cols = (axis == 2) ? y : z;
        const auto out_size = rows * cols * sizeof(double);

        // allocate memory for output tensor
        double* out_data;
        err = cudaMalloc((void**)(&out_data), out_size);
        assert(err == cudaSuccess);

        // map GPU tensor to Eigen (no copying)
        Eigen::TensorMap<tensord<2>> out_tensor(out_data, rows, cols);

        // sum along current axis
        sum_axis[0] = axis;

        // sum the tensor on GPU
        out_tensor.device(dev) = in_tensor.sum(sum_axis);
        cudaStreamSynchronize(stream.stream());

        // copy the result from GPU back to main memory
        err = cudaMemcpy(result[axis].data(), out_data, out_size,
                         cudaMemcpyDeviceToHost);
        assert(err == cudaSuccess);

        // free the last sum GPU memory
        err = cudaFree(out_data);
        assert(err == cudaSuccess);
    }

    // free input tensor GPU memory
    err = cudaFree(in_data);
    assert(err == cudaSuccess);

    return result;
}

double* cuda::apply_psi(double* begin, size_t num_elems) {
    const size_t size = num_elems * sizeof(double);

    cudaError_t err = cudaSuccess;

    // host memory wrapper
    cuda::HostMemory1D<double> host_data(begin, num_elems);

    // allocate device memory
    cuda::DeviceMemory1D<double> device_data(num_elems);

    // copy range to GPU
    cuda::copy1D(device_data, host_data, cudaMemcpyHostToDevice);

    // grid/block dimensions
    size_t threads_per_block = 1024;
    size_t blocks_per_grid =
        (num_elems + threads_per_block - 1) / threads_per_block;

    // apply kernel
    kernel::apply_psi<<<blocks_per_grid, threads_per_block>>>(
        device_data.data(), num_elems);
    err = cudaGetLastError();
    assert(err == cudaSuccess);

    // copy from GPU to main memory
    cuda::copy1D(host_data, device_data, cudaMemcpyDeviceToHost);

    return begin;
}

/**
 * @brief Return ceiling of integer division between given parameters.
 *
 * This function returns \f$\ceil{\frac{a}{b}}\f$ for parameters a and b.
 *
 * @param a Nominator.
 * @param b Denominator.
 * @return Ceiling of \f$\frac{a}{b}\f$ as an integer.
 */
static size_t int_div_ceil(size_t a, size_t b) { return a / b + (a % b != 0); }

void cuda::bld_mult::update_grad_plus(const tensord<3>& S,
                                      const matrixd& beta_eph,
                                      tensord<3>& grad_plus) {
    // tensor dimensions
    const auto x = static_cast<size_t>(S.dimension(0));
    const auto y = static_cast<size_t>(S.dimension(1));
    const auto z = static_cast<size_t>(S.dimension(2));

    cudaError_t err = cudaSuccess;

    // create host memory wrappers
    HostMemory3D<const double> host_S(S);
    HostMemory3D<double> host_grad_plus(grad_plus);
    HostMemory2D<const double> host_beta_eph(beta_eph);

    // allocate device memory
    DeviceMemory3D<double> device_S(S);
    DeviceMemory2D<double> device_beta_eph(beta_eph);
    DeviceMemory3D<double> device_grad_plus(grad_plus);

    // copy S to GPU
    copy3D(device_S, host_S, cudaMemcpyHostToDevice);

    // copy beta_eph to GPU
    copy2D(device_beta_eph, host_beta_eph, cudaMemcpyHostToDevice);

    // block dimensions (number of threads per block axis)
    constexpr size_t block_size_x = 16;
    constexpr size_t block_size_y = 16;
    constexpr size_t block_size_z = 4;
    dim3 block_dims(block_size_y, block_size_x, block_size_z);
    dim3 grid_dims(int_div_ceil(y, block_size_y), int_div_ceil(x, block_size_x),
                   int_div_ceil(z, block_size_z));

    // run kernel
    kernel::update_grad_plus<<<grid_dims, block_dims>>>(
        device_S.pitched_ptr(), device_beta_eph.data(), device_beta_eph.pitch(),
        device_grad_plus.pitched_ptr(), y, x, z);
    err = cudaGetLastError();
    assert(err == cudaSuccess);

    // copy result onto grad_plus
    copy3D(host_grad_plus, device_grad_plus, cudaMemcpyDeviceToHost);
}