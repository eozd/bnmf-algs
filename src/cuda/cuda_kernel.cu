#include <cuda_runtime.h>
#include <vector>
#include <cassert>
#include "defs.hpp"

__global__ void tensor_sum(const double* tensor, double* out, size_t axis) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
}

void cuda_init() {
    cudaError_t err = cudaSuccess;
    err = cudaSetDevice(0);
    assert(err == cudaSuccess);
}

bnmf_algs::matrix_t cuda_tensor_sum(const bnmf_algs::tensord<3>& tensor, size_t axis) {
    long x = tensor.dimension(0), y = tensor.dimension(1), z = tensor.dimension(2);
    size_t tensor_size = (x * y * z) * sizeof(double);
    cudaError_t err = cudaSuccess;

    long first_dim = axis == 0 ? y : x;
    long second_dim = axis == 2 ? y : z;
    size_t out_size = (first_dim * second_dim) * sizeof(double);

    bnmf_algs::matrix_t out(first_dim, second_dim);

    double* device_tensor = nullptr;
    err = cudaMalloc((void**)&device_tensor, tensor_size);
    assert(err == cudaSuccess);

    double* device_out = nullptr;
    err = cudaMalloc((void**)&device_out, out_size);
    assert(err == cudaSuccess);

    err = cudaMemcpy(device_tensor, tensor.data(), tensor_size, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);

    int threads_per_block = 256;
    int blocks_per_grid = (x * y + threads_per_block - 1) / threads_per_block;
    tensor_sum<<<blocks_per_grid, threads_per_block>>>(device_tensor, device_out, axis);
    err = cudaGetLastError();
    assert(err == cudaSuccess);

    err = cudaMemcpy(out.data(), device_out, out_size, cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);

    err = cudaFree(device_tensor);
    assert(err == cudaSuccess);
    err = cudaFree(device_out);
    assert(err == cudaSuccess);

    return out;
}
