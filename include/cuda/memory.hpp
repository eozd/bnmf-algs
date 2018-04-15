#pragma once

#include "defs.hpp"
#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>

namespace bnmf_algs {
namespace cuda {

template <typename T> class HostMemory1D {
  public:
    explicit HostMemory1D(T* data, size_t num_elems)
        : m_num_elems(num_elems), m_data(data){};

    T* data() const { return m_data; }
    size_t bytes() const { return m_num_elems * sizeof(T); }

  private:
    size_t m_num_elems;
    T* m_data;
};

template <typename T> class DeviceMemory1D {
  public:
    explicit DeviceMemory1D(size_t num_elems)
        : m_num_elems(num_elems), m_data(nullptr) {
        size_t alloc_size = num_elems * sizeof(T);
        cudaError_t err = cudaMalloc((void**)(&m_data), alloc_size);
        assert(err == cudaSuccess);
    };

    ~DeviceMemory1D() {
        cudaError_t err = cudaFree(m_data);
        assert(err == cudaSuccess);
    }

    T* data() const { return m_data; }
    size_t bytes() const { return m_num_elems * sizeof(T); }

  private:
    size_t m_num_elems;
    T* m_data;
};


template <typename Memory1, typename Memory2>
void copy1D(Memory1& destination, const Memory2& source, cudaMemcpyKind kind) {
    cudaError_t err =
        cudaMemcpy(destination.data(), source.data(), source.bytes(), kind);
    assert(err == cudaSuccess);
}

} // namespace cuda
} // namespace bnmf_algs
