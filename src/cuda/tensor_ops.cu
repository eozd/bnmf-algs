#include "cuda/tensor_ops.hpp"
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
    const size_t double_bytes = sizeof(double);

    // input tensor properties
    const long x = tensor.dimension(0);
    const long y = tensor.dimension(1);
    const long z = tensor.dimension(2);
    const auto in_num_elems = static_cast<size_t>(x * y * z);
    const size_t in_size = in_num_elems * double_bytes;

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
        const auto out_size = rows * cols * double_bytes;

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
