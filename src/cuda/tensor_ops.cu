#include "cuda/tensor_ops.hpp"
#include "defs.hpp"
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace bnmf_algs;

__device__ void psi_appr(double& x) {
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

    x = res - extra;
}

__global__ void apply_psi_kernel(double* begin, size_t num_elems) {
    size_t i = blockDim.x * blockIdx.y + threadIdx.x;
    if (i < num_elems) {
        psi_appr(begin[i]);
    }
}

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

    apply_psi_kernel<<<blocks_per_grid, threads_per_block>>>(gpu_data,
                                                             num_elems);
    err = cudaGetLastError();
    assert(err == cudaSuccess);

    cudaMemcpy(begin, gpu_data, size, cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);

    cudaFree(gpu_data);
    assert(err == cudaSuccess);

    return begin;
}