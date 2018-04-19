#pragma once

#include "defs.hpp"
#include <cstddef>
#include <cuda_runtime.h>

namespace bnmf_algs {
namespace cuda {
/**
 * @brief A wrapper template class around 3D row-major pitched memory stored in
 * device memory (GPU memory).
 *
 * DeviceMemory3D class represents the memory of a <b>row-major</b>
 * tensor stored in device memory (GPU memory). The intended
 * use of this class is to provide an interface that can be used with CUDA
 * functions.
 *
 * DeviceMemory3D class <b>owns the memory it allocates on the GPU</b>, and
 * therefore is responsible from the deallocation of the allocated GPU memory.
 * GPU memory allocation is performed upon object construction and deallocation
 * is performed upon object destruction.
 *
 * DeviceMemory3D objects can be used with cuda::copy3D function along with
 * HostMemory3D objects to copy memory from host/device to host/device
 * locations. See cuda::copy3D for details about copying memory using CUDA
 * functions.
 *
 * @tparam T Type of the values in the given memory address.
 */
template <typename T> class DeviceMemory3D {
  public:
    /**
     * @brief Type of the values stored on GPU.
     */
    using value_type = T;

    /**
     * @brief Construct a DeviceMemory3D object responsible from the GPU memory
     * allocated with respect to the given host matrix.
     *
     * This constructor allocates a 3D row-major pitched memory on the GPU using
     * cudaMalloc3D function. The allocated tensor memory is in row-major
     * order. Therefore, pitch of the GPU memory is set as the number of bytes
     * of a single fiber of the tensor (along 3rd dimension). See cudaMalloc3D
     * documentation to learn more about the intrinsics of the memory allocation
     * procedure.
     *
     * @param first_dim First dimension of the 3D memory to allocate.
     * @param second_dim Second dimension of the 3D memory to allocate.
     * @param third_dim Third dimension of the 3D memory to allocate. Since the
     * tensor is allocated in row-major order, allocation pitch is set as the
     * number of bytes required to store a single fiber along the third
     * dimension.
     */
    explicit DeviceMemory3D(size_t first_dim, size_t second_dim,
                            size_t third_dim)
        : m_dims(shape<3>{first_dim, second_dim, third_dim}),
          m_extent(
              make_cudaExtent(third_dim * sizeof(T), second_dim, first_dim)),
          m_ptr() {
        auto err = cudaMalloc3D(&m_ptr, m_extent);
        BNMF_ASSERT(
            err == cudaSuccess,
            "Error allocating memory in cuda::DeviceMemory3D::DeviceMemory3D");
    }

    /**
     * @brief Copy constructor (deleted).
     *
     * Since DeviceMemory3D objects own the memory they own, only move operators
     * are allowed.
     */
    DeviceMemory3D(const DeviceMemory3D&) = delete;

    /**
     * @brief Copy assignment operator (deleted).
     *
     * Since DeviceMemory3D objects own the memory they own, only move operators
     * are allowed.
     */
    DeviceMemory3D& operator=(const DeviceMemory3D&) = delete;

    /**
     * @brief Move constructor.
     *
     * Move construct a new DeviceMemory3D object. After this constructor exits,
     * the parameter DeviceMemory3D object does not point to any location and
     * therefore must not be used.
     *
     * @param other Other DeviceMemory3D object to move from.
     */
    DeviceMemory3D(DeviceMemory3D&& other)
        : m_dims(other.m_dims), m_extent(other.m_extent), m_ptr(other.m_ptr) {
        other.reset_members();
    }

    /**
     * @brief Move assignment operator.
     *
     * Move assign to this DeviceMemory3D object. After this constructor exits,
     * the parameter DeviceMemor(3D object does not point to any location and
     * therefore must not be used. Additionally, the old memory pointed by this
     * DeviceMemory3D is freed.
     *
     * @param other Other DeviceMemory3D object to move from.
     * @return Reference to assigned object.
     */
    DeviceMemory3D& operator=(DeviceMemory3D&& other) {
        this->free_cuda_mem();

        this->m_dims = other.m_dims;
        this->m_extent = other.m_extent;
        this->m_ptr = other.m_ptr;

        other.reset_members();

        return *this;
    }

    /**
     * @brief Destruct the current DeviceMemory3D object by deallocating the
     * held GPU memory.
     *
     * This function deallocates the GPU memory held by the current
     * DeviceMemory3D object. See cudaFree function documentation for memory
     * deallocation intrinsics.
     */
    ~DeviceMemory3D() { free_cuda_mem(); }

    /**
     * @brief Get the cudaPitchedPtr type storing allocation parameters and the
     * pointer to the device memory.
     *
     * Parameters of the returned pitched pointer type is set according to a
     * row-major tensor allocation. See cudaPitchedPtr type documentation for
     * the parameters provided about the allocation.
     *
     * @return cudaPitchedPtr type storing allocation parameters and the pointer
     * itself.
     */
    cudaPitchedPtr pitched_ptr() const { return m_ptr; }

    /**
     * @brief Get the cudaExtent type storing the extents of the allocation.
     *
     * Parameters of the returned extent type is set according to a row-major
     * tensor allocation. See cudaExtent type documentation for the parameters
     * provided about the allocation.
     *
     * @return cudaExtent type storing the extents of the allocation.
     */
    cudaExtent extent() const { return m_extent; }

    /**
     * @brief Get the dimensions of this memory region in terms of elements.
     *
     * @return A bnmf_algs::shape object of the form {first_dim, second_dim,
     * third_dim}.
     */
    shape<3> dims() const { return m_dims; }

  private:
    /**
     * @brief Free GPU memory.
     */
    void free_cuda_mem() {
        auto err = cudaFree(m_ptr.ptr);
        BNMF_ASSERT(
            err == cudaSuccess,
            "Error deallocating memory in cuda::DeviceMemory3D::free_cuda_mem");
    }

    /**
     * @brief Reset members.
     */
    void reset_members() {
        this->m_dims = {0, 0, 0};
        this->m_extent = {0, 0, 0};
        this->m_ptr = {nullptr, 0, 0, 0};
    }

  private:
    /**
     * @brief Dimension of this memory region as {first_dim, second_dim,
     * third_dim}.
     */
    shape<3> m_dims;

    /**
     * @brief Extents of the allocation (width, height, depth).
     */
    cudaExtent m_extent;

    /**
     * @brief Pitched pointer of 3D allocation.
     */
    cudaPitchedPtr m_ptr;
};
} // namespace cuda
} // namespace bnmf_algs
