#pragma once

#include "defs.hpp"
#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>
#include <type_traits>

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

template <typename T> class HostMemory2D {
  public:
    using nonconst_T = typename std::remove_const<T>::type;

    explicit HostMemory2D(const matrix_t<nonconst_T>& host_matrix)
        : m_data(host_matrix.data()), m_pitch(host_matrix.cols() * sizeof(T)),
          m_height(static_cast<size_t>(host_matrix.rows())){};

    T* data() const { return m_data; }
    size_t pitch() const { return m_pitch; }
    size_t width() const { return m_pitch; }
    size_t height() const { return m_height; }

  private:
    T* m_data;
    size_t m_pitch;
    size_t m_height;
};

template <typename T> class DeviceMemory2D {
  public:
    explicit DeviceMemory2D(const matrix_t<T>& host_matrix)
        : m_data(nullptr), m_pitch(), m_width(host_matrix.cols() * sizeof(T)),
          m_height(static_cast<size_t>(host_matrix.rows())) {
        size_t rows = static_cast<size_t>(host_matrix.rows());
        size_t cols = static_cast<size_t>(host_matrix.cols());

        cudaError_t err = cudaMallocPitch((void**)(&m_data), &m_pitch,
                                          cols * sizeof(T), rows);
        assert(err == cudaSuccess);
    };

    ~DeviceMemory2D() {
        cudaError_t err = cudaFree(m_data);
        assert(err == cudaSuccess);
    }

    T* data() const { return m_data; }
    size_t pitch() const { return m_pitch; }
    size_t width() const { return m_width; }
    size_t height() const { return m_height; }

  private:
    T* m_data;
    size_t m_pitch;
    size_t m_width;
    size_t m_height;
};

template <typename T> class HostMemory3D {
  public:
    using nonconst_T = typename std::remove_const<T>::type;

    explicit HostMemory3D(const tensor_t<nonconst_T, 3>& host_tensor)
        : m_extent(
              make_cudaExtent(host_tensor.dimension(2) * sizeof(T),
                              static_cast<size_t>(host_tensor.dimension(1)),
                              static_cast<size_t>(host_tensor.dimension(0)))),
          m_ptr() {
        m_ptr.pitch = host_tensor.dimension(2) * sizeof(T);
        m_ptr.xsize = static_cast<size_t>(host_tensor.dimension(2));
        m_ptr.ysize = static_cast<size_t>(host_tensor.dimension(1));
        m_ptr.ptr = (void*)(host_tensor.data());
    }

    cudaPitchedPtr pitched_ptr() const { return m_ptr; }
    cudaExtent extent() const { return m_extent; }

  private:
    cudaExtent m_extent;
    cudaPitchedPtr m_ptr;
};

template <typename T> class DeviceMemory3D {
  public:
    explicit DeviceMemory3D(const tensor_t<T, 3>& host_tensor)
        : m_extent(
              make_cudaExtent(host_tensor.dimension(2) * sizeof(T),
                              static_cast<size_t>(host_tensor.dimension(1)),
                              static_cast<size_t>(host_tensor.dimension(0)))),
          m_ptr() {
        cudaError_t err = cudaMalloc3D(&m_ptr, m_extent);
        assert(err == cudaSuccess);
    }

    ~DeviceMemory3D() {
        cudaError_t err = cudaFree(m_ptr.ptr);
        assert(err == cudaSuccess);
    }

    cudaPitchedPtr pitched_ptr() const { return m_ptr; }
    cudaExtent extent() const { return m_extent; }

  private:
    cudaExtent m_extent;
    cudaPitchedPtr m_ptr;
};

template <typename DstMemory1D, typename SrcMemory1D>
void copy1D(DstMemory1D& destination, const SrcMemory1D& source,
            cudaMemcpyKind kind) {
    cudaError_t err =
        cudaMemcpy(destination.data(), source.data(), source.bytes(), kind);
    assert(err == cudaSuccess);
}

template <typename DstMemory2D, typename SrcMemory2D>
void copy2D(DstMemory2D& destination, const SrcMemory2D& source,
            cudaMemcpyKind kind) {
    cudaError_t err =
        cudaMemcpy2D(destination.data(), destination.pitch(), source.data(),
                     source.pitch(), source.width(), source.height(), kind);
    assert(err == cudaSuccess);
}

template <typename DstMemory3D, typename SrcMemory3D>
void copy3D(DstMemory3D& destination, const SrcMemory3D& source,
            cudaMemcpyKind kind) {
    // send S tensor to GPU
    cudaMemcpy3DParms params = {0};
    params.srcPtr = source.pitched_ptr();
    params.dstPtr = destination.pitched_ptr();
    params.extent = source.extent();
    params.kind = kind;

    cudaError_t err = cudaMemcpy3D(&params);
    assert(err == cudaSuccess);
}
} // namespace cuda
} // namespace bnmf_algs
