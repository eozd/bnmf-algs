#include "cuda/tensor_ops.hpp"
#include "cuda/memory.hpp"
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
    kernel::apply_psi<<<blocks_per_grid, threads_per_block>>>(device_data.data(),
                                                              num_elems);
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
    auto x = static_cast<size_t>(S.dimension(0));
    auto y = static_cast<size_t>(S.dimension(1));
    auto z = static_cast<size_t>(S.dimension(2));

    cudaError_t err = cudaSuccess;

    // make tensor extent object. Since we store tensors in row-major order,
    // width of the allocation is the number of bytes of a single fiber.
    const auto tensor_extent = make_cudaExtent(z * sizeof(double), y, x);

    // allocate memory for S
    cudaPitchedPtr S_gpu;
    err = cudaMalloc3D(&S_gpu, tensor_extent);
    assert(err == cudaSuccess);

    // allocate memory for beta_eph
    double* beta_eph_gpu;
    size_t beta_eph_pitch;
    err = cudaMallocPitch((void**)(&beta_eph_gpu), &beta_eph_pitch,
                          z * sizeof(double), y);
    assert(err == cudaSuccess);

    // allocate memory for grad_plus
    cudaPitchedPtr grad_plus_gpu;
    err = cudaMalloc3D(&grad_plus_gpu, tensor_extent);
    assert(err == cudaSuccess);

    // send S tensor to GPU
    cudaMemcpy3DParms params_3D = {0};
    params_3D.srcPtr.ptr = (void*)S.data();
    params_3D.srcPtr.pitch = S.dimension(2) * sizeof(double); // row major
    params_3D.srcPtr.xsize = S.dimension(2);
    params_3D.srcPtr.ysize = S.dimension(1);
    params_3D.dstPtr = S_gpu;
    params_3D.extent = tensor_extent;
    params_3D.kind = cudaMemcpyHostToDevice;

    err = cudaMemcpy3D(&params_3D);
    assert(err == cudaSuccess);

    // send beta_eph matrix to GPU
    err = cudaMemcpy2D(beta_eph_gpu, beta_eph_pitch, beta_eph.data(),
                       z * sizeof(double), z * sizeof(double), y,
                       cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);

    // block dimensions (number of threads per block axis)
    constexpr size_t block_size_x = 16;
    constexpr size_t block_size_y = 16;
    constexpr size_t block_size_z = 4;
    dim3 block_dims(block_size_y, block_size_x, block_size_z);
    dim3 grid_dims(int_div_ceil(y, block_size_y), int_div_ceil(x, block_size_x),
                   int_div_ceil(z, block_size_z));

    // run kernel
    kernel::update_grad_plus<<<grid_dims, block_dims>>>(
        S_gpu, beta_eph_gpu, beta_eph_pitch, grad_plus_gpu, y, x, z);
    err = cudaGetLastError();
    assert(err == cudaSuccess);

    // copy from GPU onto grad_plus tensor
    params_3D.srcPtr = grad_plus_gpu;
    params_3D.dstPtr.ptr = grad_plus.data();
    params_3D.dstPtr.pitch = grad_plus.dimension(2) * sizeof(double);
    params_3D.dstPtr.xsize = grad_plus.dimension(2);
    params_3D.dstPtr.ysize = grad_plus.dimension(1);
    params_3D.extent = tensor_extent;
    params_3D.kind = cudaMemcpyDeviceToHost;

    err = cudaMemcpy3D(&params_3D);
    assert(err == cudaSuccess);

    // free memory
    err = cudaFree(S_gpu.ptr);
    assert(err == cudaSuccess);
    err = cudaFree(beta_eph_gpu);
    assert(err == cudaSuccess);
    err = cudaFree(grad_plus_gpu.ptr);
    assert(err == cudaSuccess);
}