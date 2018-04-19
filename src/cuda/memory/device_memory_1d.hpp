#pragma once

#include "defs.hpp"
#include <cstddef>
#include <cuda_runtime.h>

namespace bnmf_algs {
namespace cuda {

/**
 * @brief A wrapper template class around a contiguous array of T types laid out
 * in device memory (GPU memory).
 *
 * DeviceMemory1D class represents the memory for a contiguous sequence of T
 * values in device memory. The intended use of this class is to provide an
 * interface that can be used with CUDA functions.
 *
 * DeviceMemory1D class <b>owns the device memory it allocates</b>, and
 * therefore is responsible from the deallocation of the allocated GPU memory.
 * This operation is performed upon object destruction.
 *
 * DeviceMemory1D objects can be used with cuda::copy1D function along with
 * HostMemory1D objects to copy memory from host/device to host/device
 * locations. See cuda::copy1D for details about copying memory using CUDA
 * functions.
 *
 * @tparam T Type of the values to be allocated in GPU device.
 */
template <typename T> class DeviceMemory1D {
  public:
    /**
     * @brief Type of the values wrapped around current DeviceMemory1D
     * object.
     */
    using value_type = T;

    /**
     * @brief Construct a DeviceMemory1D object that allocates num_elems many
     * T values on the GPU device.
     *
     * After this DeviceMemory1D object is constructed, there will be num_elems
     * many T types allocated on the GPU device. See cudaMalloc function
     * documentation for memory allocation intrinsics.
     *
     * @param num_elems Number of T types to allocate on the GPU.
     */
    explicit DeviceMemory1D(size_t num_elems)
        : m_dims(shape<1>{num_elems}), m_data(nullptr) {
        size_t alloc_size = num_elems * sizeof(T);
        auto err = cudaMalloc((void**)(&m_data), alloc_size);
        BNMF_ASSERT(
            err == cudaSuccess,
            "Error allocating memory in cuda::DeviceMemory1D::DeviceMemory1D");
    };

    /**
     * @brief Copy constructor (deleted).
     *
     * Since DeviceMemory1D objects own the memory they own, only move operators
     * are allowed.
     */
    DeviceMemory1D(const DeviceMemory1D&) = delete;

    /**
     * @brief Copy assignment operator (deleted).
     *
     * Since DeviceMemory1D objects own the memory they own, only move operators
     * are allowed.
     */
    DeviceMemory1D& operator=(const DeviceMemory1D&) = delete;

    /**
     * @brief Move constructor.
     *
     * Move construct a new DeviceMemory1D object. After this constructor exits,
     * the parameter DeviceMemory1D object does not point to any location and
     * therefore must not be used.
     *
     * @param other Other DeviceMemory1D object to move from.
     */
    DeviceMemory1D(DeviceMemory1D&& other)
        : m_dims(other.m_dims), m_data(other.m_data) {
        other.reset_members();
    }

    /**
     * @brief Move assignment operator.
     *
     * Move assign to this DeviceMemory1D object. After this constructor exits,
     * the parameter DeviceMemor(1D object does not point to any location and
     * therefore must not be used. Additionally, the old memory pointed by this
     * DeviceMemory1D is freed.
     *
     * @param other Other DeviceMemory1D object to move from.
     * @return Reference to assigned object.
     */
    DeviceMemory1D& operator=(DeviceMemory1D&& other) {
        this->free_cuda_mem();
        this->m_dims = other.m_dims;
        this->m_data = other.m_data;

        other.reset_members();

        return *this;
    }

    /**
     * @brief Destruct the current DeviceMemory1D object by deallocating the
     * held GPU memory.
     *
     * This function deallocates the GPU memory held by the current
     * DeviceMemory1D object. See cudaFree function documentation for memory
     * deallocation intrinsics.
     */
    ~DeviceMemory1D() { free_cuda_mem(); }

    /**
     * @brief Get a device pointer pointing to the memory allocated by the
     * current DeviceMemory1D object.
     *
     * @return Device pointer pointing to the allocated GPU memory.
     */
    T* data() const { return m_data; }

    /**
     * @brief Get the number of bytes of the allocated memory on the GPU.
     *
     * @return Number of bytes of the allocated GPU memory.
     */
    size_t bytes() const { return m_dims[0] * sizeof(T); }

    /**
     * @brief Get the dimensions of this memory region in terms of elements.
     *
     * @return A bnmf_algs::shape representing the dimension.
     */
    shape<1> dims() const { return m_dims; }

  private:
    /**
     * @brief Free the GPU memory pointed by m_data.
     */
    void free_cuda_mem() {
        auto err = cudaFree(m_data);
        BNMF_ASSERT(
            err == cudaSuccess,
            "Error deallocating memory in cuda::DeviceMemory1D::free_cuda_mem");
    }

    /**
     * @brief Reset all members.
     */
    void reset_members() {
        m_dims[0] = 0;
        m_data = nullptr;
    }

  private:
    /**
     * @brief Dimension (length) of this memory region.
     */
    shape<1> m_dims;

    /**
     * @brief Device pointer pointing to the beginning address of the GPU memory
     * sequence.
     */
    T* m_data;
};
} // namespace cuda
} // namespace bnmf_algs
