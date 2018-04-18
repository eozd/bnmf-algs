#include "defs.hpp"
#include <cstddef>
#include <cuda_runtime.h>

namespace bnmf_algs {
namespace cuda {
/**
 * @brief A wrapper template class around 2D row-major pitched memory stored in
 * device memory (GPU memory).
 *
 * DeviceMemory2D class represents the memory of a <b>row-major</b>
 * matrix type stored in device memory (GPU memory). The intended
 * use of this class is to provide an interface that can be used with CUDA
 * functions.
 *
 * DeviceMemory2D class <b>owns the memory it allocates on the GPU</b>, and
 * therefore is responsible from the deallocation of the allocated GPU memory.
 * GPU memory allocation is performed upon object construction and deallocation
 * is performed upon object destruction.
 *
 * DeviceMemory2D objects can be used with cuda::copy2D function along with
 * HostMemory2D objects to copy memory from host/device to host/device
 * locations. See cuda::copy2D for details about copying memory using CUDA
 * functions.
 *
 * @tparam T Type of the values in the given memory address.
 */
template <typename T> class DeviceMemory2D {
  public:
    /**
     * @brief Type of the values stored on GPU.
     */
    using value_type = T;

    /**
     * @brief Construct a DeviceMemory2D object responsible from the GPU memory
     * allocated with respect to the given host matrix.
     *
     * This constructor allocates a 2D pitched memory on the GPU using
     * cudaMallocPitch function. The allocated matrix memory is in row-major
     * order. Therefore, pitch of the GPU memory is set as the number of bytes
     * of a single row of the matrix. See cudaMallocPitch documentation to learn
     * more about the intrinsics of the memory allocation procedure.
     *
     * @param rows Number of rows of the row-major GPU matrix to allocate.
     * @param cols Number of columns of the row-major GPU matrix to allocate.
     */
    explicit DeviceMemory2D(size_t rows, size_t cols)
        : m_data(nullptr), m_pitch(), m_dims(shape<2>{rows, cols}) {
        auto err = cudaMallocPitch((void**)(&m_data), &m_pitch,
                                   cols * sizeof(T), rows);
        BNMF_ASSERT(
            err == cudaSuccess,
            "Error allocating memory in cuda::DeviceMemory2D::DeviceMemory2D");
    };

    /**
     * @brief Copy constructor (deleted).
     *
     * Since DeviceMemory2D objects own the memory they own, only move operators
     * are allowed.
     */
    DeviceMemory2D(const DeviceMemory2D&) = delete;

    /**
     * @brief Copy assignment operator (deleted).
     *
     * Since DeviceMemory2D objects own the memory they own, only move operators
     * are allowed.
     */
    DeviceMemory2D& operator=(const DeviceMemory2D&) = delete;

    /**
     * @brief Move constructor.
     *
     * Move construct a new DeviceMemory2D object. After this constructor exits,
     * the parameter DeviceMemory2D object does not point to any location and
     * therefore must not be used.
     *
     * @param other Other DeviceMemory2D object to move from.
     */
    DeviceMemory2D(DeviceMemory2D&& other)
        : m_data(other.m_data), m_pitch(other.m_pitch), m_dims(other.m_dims) {
        other.reset_members();
    }

    /**
     * @brief Move assignment operator.
     *
     * Move assign to this DeviceMemory2D object. After this constructor exits,
     * the parameter DeviceMemor(2D object does not point to any location and
     * therefore must not be used. Additionally, the old memory pointed by this
     * DeviceMemory2D is freed.
     *
     * @param other Other DeviceMemory2D object to move from.
     * @return Reference to assigned object.
     */
    DeviceMemory2D& operator=(DeviceMemory2D&& other) {
        this->free_cuda_mem();

        this->m_data = other.m_data;
        this->m_pitch = other.m_pitch;
        this->m_dims = other.m_dims;

        other.reset_members();

        return *this;
    }

    /**
     * @brief Destruct the current DeviceMemory2D object by deallocating the
     * held GPU memory.
     *
     * This function deallocates the GPU memory held by the current
     * DeviceMemory2D object. See cudaFree function documentation for memory
     * deallocation intrinsics.
     */
    ~DeviceMemory2D() { free_cuda_mem(); }

    /**
     * @brief Get a device pointer pointing to the pitched memory allocated by
     * the current DeviceMemory2D object.
     *
     * @return Device pointer pointing to the allocated GPU memory.
     */
    T* data() const { return m_data; }

    /**
     * @brief Get the pitch of the allocation.
     *
     * Pitch of the allocation is defined as the number of bytes required to
     * store a single row of the matrix, including the padding bytes.
     *
     * @return Pitch (number of bytes of a single row including padding bytes).
     */
    size_t pitch() const { return m_pitch; }

    /**
     * @brief Get the width of the allocation in terms of bytes.
     *
     * Width of the allocation is defined as the number of bytes required to
     * store a single row of the matrix, excluding the padding bytes. Note that
     * DeviceMemory2D pitch and width member function generally give different
     * results due to the padding bytes used by cudaMallocPitch function.
     *
     * @return Width (number of bytes of a single row excluding padding bytes).
     */
    size_t width() const { return m_dims[1] * sizeof(T); }

    /**
     * @brief Get the height of the allocation in terms of number of elements.
     *
     * Height of the allocation is defined as the number of elements in a single
     * column of the allocated matrix.
     *
     * @return Height (number of elements in a single column of the matrix).
     */
    size_t height() const { return m_dims[0]; }

    /**
     * @brief Get the dimensions of this memory region in terms of elements.
     *
     * @return A bnmf_algs::shape<2> object of the form {rows, cols}.
     */
    shape<2> dims() const { return m_dims; }

  private:
    /**
     * @brief Free GPU memory.
     */
    void free_cuda_mem() {
        auto err = cudaFree(m_data);
        BNMF_ASSERT(
            err == cudaSuccess,
            "Error deallocating memory in cuda::DeviceMemory2D::free_cuda_mem");
    }

    /**
     * @brief Reset all members.
     */
    void reset_members() {
        this->m_data = nullptr;
        this->m_pitch = 0;
        this->m_dims = {0, 0};
    }

  private:
    /**
     * @brief Device pointer pointing to the beginning address of the GPU memory
     * storing the matrix.
     */
    T* m_data;

    /**
     * @brief Pitch of the allocation (number of bytes of a single row including
     * padding bytes).
     */
    size_t m_pitch;

    /**
     * @brief Dimensions of this memory region as {rows, cols}.
     */
    shape<2> m_dims;
};
} // namespace cuda
} // namespace bnmf_algs
