#include "cuda/tensor_ops.hpp"
#include "defs.hpp"
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace bnmf_algs;

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

    // create GPU stream and device
    Eigen::CudaStreamDevice stream;
    Eigen::GpuDevice dev(&stream);

    // sum axis
    Eigen::array<size_t, 1> sum_axis;

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

    double* gpu_data;
    err = cudaMalloc((void**)(&gpu_data), size);
    assert(err == cudaSuccess);

    cudaMemcpy(gpu_data, begin, size, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);

    size_t threads_per_block = 1024;
    size_t blocks_per_grid =
        (num_elems + threads_per_block - 1) / threads_per_block;

    kernel::apply_psi<<<blocks_per_grid, threads_per_block>>>(gpu_data,
                                                              num_elems);
    err = cudaGetLastError();
    assert(err == cudaSuccess);

    cudaMemcpy(begin, gpu_data, size, cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);

    cudaFree(gpu_data);
    assert(err == cudaSuccess);

    return begin;
}

static size_t int_div_ceil(size_t a, size_t b) { return a / b + (a % b != 0); }

void cuda::bld_mult::update_grad_plus(const tensord<3>& S,
                                      const matrix_t& beta_eph,
                                      tensord<3>& grad_plus) {
    cudaError_t err;
    auto x = static_cast<size_t>(S.dimension(0));
    auto y = static_cast<size_t>(S.dimension(1));
    auto z = static_cast<size_t>(S.dimension(2));

    const auto tensor_extent = make_cudaExtent(z * sizeof(double), y, x);

    // allocate memory for S
    cudaPitchedPtr S_gpu;
    err = cudaMalloc3D(&S_gpu, tensor_extent);
    assert(err == cudaSuccess);

    // allocate memory for beta_eph
    double* beta_eph_gpu;
    size_t pitch;
    err =
        cudaMallocPitch((void**)(&beta_eph_gpu), &pitch, z * sizeof(double), y);
    assert(err == cudaSuccess);

    // allocate memory for grad_plus
    cudaPitchedPtr grad_plus_gpu;
    err = cudaMalloc3D(&grad_plus_gpu, tensor_extent);
    assert(err == cudaSuccess);

    // send S tensor to GPU
    cudaMemcpy3DParms params_3D = {0};
    params_3D.srcPtr.ptr = (void*)S.data();
    params_3D.srcPtr.pitch = S.dimension(2) * sizeof(double);
    params_3D.srcPtr.xsize = S.dimension(2);
    params_3D.srcPtr.ysize = S.dimension(1);
    params_3D.dstPtr = S_gpu;
    params_3D.extent = tensor_extent;
    params_3D.kind = cudaMemcpyHostToDevice;

    err = cudaMemcpy3D(&params_3D);
    assert(err == cudaSuccess);

    // send beta_eph matrix to GPU
    err = cudaMemcpy2D(beta_eph_gpu, pitch, beta_eph.data(), z * sizeof(double),
                       z * sizeof(double), y, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);

    // dimensions
    constexpr size_t block_size_x = 16;
    constexpr size_t block_size_y = 16;
    constexpr size_t block_size_z = 4;
    dim3 block_dims(block_size_y, block_size_x, block_size_z);
    dim3 grid_dims(int_div_ceil(y, block_size_y), int_div_ceil(x, block_size_x),
                   int_div_ceil(z, block_size_z));

    // run kernel
    kernel::update_grad_plus<<<grid_dims, block_dims>>>(
        S_gpu, beta_eph_gpu, pitch, grad_plus_gpu, y, x, z);
    err = cudaGetLastError();
    assert(err == cudaSuccess);

    // copy from GPU to grad_plus tensor
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