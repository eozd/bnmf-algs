#pragma once

#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>
#include <type_traits>

namespace bnmf_algs {
namespace cuda {

/**
 * @brief A wrapper template class around a contiguous array of T types laid out
 * in main memory (host memory).
 *
 * HostMemory1D class represents the memory for a contiguous sequence of T
 * values in host memory. The intended use of this class is to provide an
 * interface that can be used with CUDA functions.
 *
 * HostMemory1D class <b>does not own the memory it is given</b>. Therefore,
 * no allocation, copying or memory freeing is performed. The only use case of
 * HostMemory1D is to provide a unified interface with DeviceMemory1D so that
 * the two classes can be used interchangeably by cuda::copy1D function. See
 * cuda::copy1D for details about copying memory using CUDA functions from
 * host/device to host/device memory.
 *
 * @tparam T Type of the values in the given memory address.
 */
template <typename T> class HostMemory1D {
  public:
    /**
     * @brief Type of the values stored in the memory sequence wrapped
     * around the current HostMemory1D object.
     */
    using value_type = T;

    /**
     * @brief Construct a HostMemory1D class around the memory given by address
     * and number of elements.
     *
     * The given memory address is assumed to reside in host memory (main
     * memory). Therefore, this function does not perform any memory allocation
     * on main memory or GPU device.
     *
     * @param data Beginning address of the sequence of T values.
     * @param num_elems Number of T values in the sequence starting at parameter
     * data.
     */
    HostMemory1D(T* data, size_t num_elems)
        : m_num_elems(num_elems), m_data(data){};

    /**
     * @brief Get the address of the memory sequence wrapped with the current
     * HostMemory1D object.
     *
     * @return Beginning address of the memory sequence wrapped with the
     * current HostMemory1D object.
     */
    T* data() const { return m_data; }

    /**
     * @brief Get the number of bytes of the memory sequence wrapped with the
     * current HostMemory1D object.
     *
     * @return Number of bytes of the memory sequence.
     */
    size_t bytes() const { return m_num_elems * sizeof(T); }

  private:
    /**
     * @brief Number of elements (T values) in the memory sequence starting at
     * m_data.
     */
    size_t m_num_elems;
    /**
     * @brief Beginning address of the memory sequence.
     */
    T* m_data;
};

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
        : m_num_elems(num_elems), m_data(nullptr) {
        size_t alloc_size = num_elems * sizeof(T);
        cudaError_t err = cudaMalloc((void**)(&m_data), alloc_size);
        assert(err == cudaSuccess);
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
        : m_num_elems(other.m_num_elems), m_data(other.m_data) {
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
        this->m_num_elems = other.m_num_elems;
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
    size_t bytes() const { return m_num_elems * sizeof(T); }

  private:
    /**
     * @brief Free the GPU memory pointed by m_data.
     */
    void free_cuda_mem() {
        cudaError_t err = cudaFree(m_data);
        assert(err == cudaSuccess);
    }

    /**
     * @brief Reset all members.
     */
    void reset_members() {
        m_num_elems = 0;
        m_data = nullptr;
    }

  private:
    /**
     * @brief Number of elements in the GPU memory sequence.
     */
    size_t m_num_elems;

    /**
     * @brief Device pointer pointing to the beginning address of the GPU memory
     * sequence.
     */
    T* m_data;
};

/**
 * @brief A wrapper template class around a row-major matrix type stored in main
 * memory (host memory).
 *
 * HostMemory2D class represents the memory of a <b>row-major</b>
 * matrix type stored in main memory (host memory). The intended
 * use of this class is to provide an interface that can be used with CUDA
 * functions.
 *
 * HostMemory2D class <b>does not own the memory it is given</b>. Therefore,
 * no allocation, copying or memory freeing is performed. The only use case of
 * HostMemory2D is to provide a unified interface with DeviceMemory2D so that
 * the two classes can be used interchangeably by cuda::copy2D function. See
 * cuda::copy2D for details about copying 2D memory using CUDA functions from
 * host/device to host/device memory.
 *
 * If the pointer pointing to the memory is const, then
 * the type of this class must be marked as const to prevent illegal mutating
 * code from accessing the memory. For example,
 * @code
 *     const matrix_t<int> mat(5, 3);
 *
 *     // mark T as const so that the instantiations of the template functions
 *     // and the member pointers are marked as const.
 *     HostMemory2D<const int> host_memory(mat.data(), mat.rows(), mat.cols());
 * @endcode
 *
 * @tparam T Type of the values in the given memory address.
 */
template <typename T> class HostMemory2D {
  public:
    /**
     * @brief Type of the values wrapped around current DeviceMemory1D
     * object.
     */
    using value_type = T;

    /**
     * @brief Construct a HostMemory2D class around the memory given by the
     * pointer and the rows and columns of the matrix.
     *
     * The memory given by pointer is assumed to reside in main memory.
     * Therefore, this function does not perform any memory allocation on
     * main memory or GPU device.
     *
     * @param data Address of the beginning of the memory storing the row-major
     * 2D matrix.
     * @param rows Number of rows of the matrix.
     * @param cols Number of columns of the matrix.
     */
    explicit HostMemory2D(T* data, size_t rows, size_t cols)
        : m_data(data), m_pitch(cols * sizeof(T)), m_height(rows){};

    /**
     * @brief Get the beginning address of the 2D memory wrapped by this
     * HostMemory2D object.
     *
     * @return Pointer to the beginning of the 2D matrix memory.
     */
    T* data() const { return m_data; }

    /**
     * @brief Get the pitch of the 2D matrix memory wrapped by this HostMemory2D
     * object.
     *
     * Pitch of the allocation is defined as the number of bytes of a single
     * row of the 2D matrix memory, including the padding bytes, <b>stored in
     * row-major order</b>.
     *
     * @return Pitch (number of bytes of a single row, including padding bytes,
     * of the matrix).
     */
    size_t pitch() const { return m_pitch; }

    /**
     * @brief Get the width of the 2D matrix <b>in terms of bytes</b>.
     *
     * Width is defined as the number of bytes of a single row of the 2D matrix
     * memory, without the padding bytes, <b>stored in row-major order</b>.
     *
     * The fact that width member returning a value in terms of bytes whereas
     * height returning in terms of number of elements may seem weird to new
     * users of this API. However, this is the convention adopted by CUDA
     * functions. Since this library tries to be as compatible with CUDA
     * function usage conventions as possible, we follow the same pattern and
     * return the width of a matrix in terms of bytes.
     *
     * <b>It is assumed that the 2D matrix stored in the main memory does not
     * use any padding bytes</b>. Therefore, the results of width and pitch
     * member functions are the same for host memory matrices.
     *
     * @return Width (number of bytes of a single row, excluding padding bytes,
     * of the matrix).
     */
    size_t width() const { return m_pitch; }

    /**
     * @brief Get the height of the 2D matrix <b>in terms of number of
     * elements</b>.
     *
     * Height is defined as the number of elements in a single column of the
     * matrix.
     *
     * @return Height (number of elements in a single column of the matrix).
     */
    size_t height() const { return m_height; }

  private:
    /**
     * @brief Host pointer pointing to the beginning of the 2D matrix memory.
     */
    T* m_data;

    /**
     * @brief Pitch of 2D matrix memory (number of bytes of a single row
     * including padding bytes).
     */
    size_t m_pitch;

    /**
     * @brief Height of the 2D matrix memory (number of elements in a single
     * column).
     */
    size_t m_height;
};

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
        : m_data(nullptr), m_pitch(), m_width(cols * sizeof(T)),
          m_height(rows) {
        cudaError_t err = cudaMallocPitch((void**)(&m_data), &m_pitch,
                                          cols * sizeof(T), rows);
        assert(err == cudaSuccess);
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
        : m_data(other.m_data), m_pitch(other.m_pitch), m_width(other.m_width),
          m_height(other.m_height) {
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
        this->m_width = other.m_width;
        this->m_height = other.m_height;

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
    size_t width() const { return m_width; }

    /**
     * @brief Get the height of the allocation in terms of number of elements.
     *
     * Height of the allocation is defined as the number of elements in a single
     * column of the allocated matrix.
     *
     * @return Height (number of elements in a single column of the matrix).
     */
    size_t height() const { return m_height; }

  private:
    /**
     * @brief Free GPU memory.
     */
    void free_cuda_mem() {
        cudaError_t err = cudaFree(m_data);
        assert(err == cudaSuccess);
    }

    /**
     * @brief Reset all members.
     */
    void reset_members() {
        this->m_data = nullptr;
        this->m_pitch = 0;
        this->m_width = 0;
        this->m_height = 0;
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
     * @brief Width of the allocation (number of bytes of a single row excluding
     * padding bytes).
     */
    size_t m_width;

    /**
     * @brief Height of the allocation (number of elements in a single column of
     * the allocated matrix).
     */
    size_t m_height;
};

/**
 * @brief A wrapper template class around a row-major 3D tensor stored in main
 * memory (host memory).
 *
 * HostMemory3D class represents the memory of a <b>row-major</b> 3D tensor type
 * stored in main memory (host memory). The intended use of this class is to
 * provide an interface that can be used with CUDA functions.
 *
 * HostMemory3D class <b>does not own the memory it is given</b>. Therefore,
 * no allocation, copying or memory freeing is performed. The only use case of
 * HostMemory3D is to provide a unified interface with DeviceMemory3D so that
 * the two classes can be used interchangeably by cuda::copy3D function. See
 * cuda::copy3D for details about copying 3D memory using CUDA functions from
 * host/device to host/device memory.
 *
 * If the pointer pointing to the memory is const, then
 * the type of this class must be marked as const to prevent illegal mutating
 * code from accessing the memory. For example,
 * @code
 *     const tensord<3> tensor(2, 3, 4);
 *
 *     // mark T as const so that the instantiations of the template functions
 *     // and the member pointers are marked as const.
 *     HostMemory3D<const double> host_memory(tensor.data(), 2, 3, 4);
 * @endcode
 *
 * @tparam T Type of the values in the given memory address.
 */
template <typename T> class HostMemory3D {
  public:
    /**
     * @brief Type of the values stored in main memory.
     */
    using value_type = T;

    /**
     * @brief Construct a HostMemory3D class around the memory given by the
     * pointer and the dimensions of a 3D row-major tensor.
     *
     * The memory given by pointer is assumed to reside in main memory.
     * Therefore, this function does not perform any memory allocation on
     * main memory or GPU device.
     *
     * @param data Address of the beginning of the memory storing the row-major
     * 3D tensor.
     * @param first_dim First dimension of the tensor. Since the given tensor
     * memory is row-major, this is the dimension whose index changes the
     * slowest when traversing the contiguous memory pointed by pointer
     * parameter data.
     * @param second_dim Second dimension of the tensor.
     * @param third_dim Third dimension of the tensor. Since the given tensor
     * memory is row-major, this is the dimension whose index changse the
     * fastest when traversing the contiguous memory pointed by pointer
     * parameter data.
     */
    explicit HostMemory3D(T* data, size_t first_dim, size_t second_dim,
                          size_t third_dim)
        : m_extent(
              make_cudaExtent(third_dim * sizeof(T), second_dim, first_dim)),
          m_ptr() {
        m_ptr.pitch = third_dim * sizeof(T);
        m_ptr.xsize = third_dim;
        m_ptr.ysize = second_dim;
        m_ptr.ptr = (void*)(data);
    }

    /**
     * @brief Get the cudaPitchedPtr type storing allocation parameters and the
     * pointer to the host memory.
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

  private:
    /**
     * @brief Extents of the allocation (width, height, depth).
     */
    cudaExtent m_extent;

    /**
     * @brief Pitched pointer of 3D allocation.
     */
    cudaPitchedPtr m_ptr;
};

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
        : m_extent(
              make_cudaExtent(third_dim * sizeof(T), second_dim, first_dim)),
          m_ptr() {
        cudaError_t err = cudaMalloc3D(&m_ptr, m_extent);
        assert(err == cudaSuccess);
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
        : m_extent(other.m_extent), m_ptr(other.m_ptr) {
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

  private:
    /**
     * @brief Free GPU memory.
     */
    void free_cuda_mem() {
        cudaError_t err = cudaFree(m_ptr.ptr);
        assert(err == cudaSuccess);
    }

    /**
     * @brief Reset members.
     */
    void reset_members() {
        this->m_extent = {0, 0, 0};
        this->m_ptr = {nullptr, 0, 0, 0};
    }

  private:
    /**
     * @brief Extents of the allocation (width, height, depth).
     */
    cudaExtent m_extent;

    /**
     * @brief Pitched pointer of 3D allocation.
     */
    cudaPitchedPtr m_ptr;
};

/**
 * @brief Copy a contiguous 1D memory from a host/device memory to a host/device
 * memory using CUDA function cudaMemcpy.
 *
 * This function copies the memory wrapped around a HostMemory1D or
 * DeviceMemory1D object to the memory wrapped around a HostMemory1D or
 * DeviceMemory1D. The type of the memory copying is given by cudaMemcpyKind
 * enum which can be one of cudaMemcpyHostToDevice, cudaMemcpyHostToHost,
 * cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice.
 *
 * See cudaMemcpy function documentation to learn more about the memory copying
 * procedure intrinsics.
 *
 * @tparam DstMemory1D Type of the destination memory. See HostMemory1D and
 * DeviceMemory1D.
 * @tparam SrcMemory1D Type of the source memory. See HostMemory1D and
 * DeviceMemory1D.
 * @param destination Destination memory object.
 * @param source Source memory object.
 * @param kind Kind of the copying to be performed.
 *
 * @throws If the copying procedure is not successful, an assertion error is
 * thrown.
 */
template <typename DstMemory1D, typename SrcMemory1D>
void copy1D(DstMemory1D& destination, const SrcMemory1D& source,
            cudaMemcpyKind kind) {
    cudaError_t err =
        cudaMemcpy(destination.data(), source.data(), source.bytes(), kind);
    assert(err == cudaSuccess);
}

/**
 * @brief Copy a contiguous 2D pitched memory from a host/device memory to a
 * host/device memory using CUDA function cudaMemcpy2D.
 *
 * This function copies the memory wrapped around a HostMemory2D or
 * DeviceMemory2D object to the memory wrapped around a HostMemory2D or
 * DeviceMemory2D. The type of the memory copying is given by cudaMemcpyKind
 * enum which can be one of cudaMemcpyHostToDevice, cudaMemcpyHostToHost,
 * cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice.
 *
 * See cudaMemcpy function documentation to learn more about the memory copying
 * procedure intrinsics.
 *
 * @tparam DstPitchedMemory2D Type of the destination memory. See HostMemory2D
 * and DeviceMemory2D.
 * @tparam SrcPitchedMemory2D Type of the source memory. See HostMemory2D and
 * DeviceMemory2D.
 * @param destination Destination memory object.
 * @param source Source memory object.
 * @param kind Kind of the copying to be performed
 *
 * @throws If the copying procedure is not successful, an assertion error is
 * thrown.
 */
template <typename DstPitchedMemory2D, typename SrcPitchedMemory2D>
void copy2D(DstPitchedMemory2D& destination, const SrcPitchedMemory2D& source,
            cudaMemcpyKind kind) {
    cudaError_t err =
        cudaMemcpy2D(destination.data(), destination.pitch(), source.data(),
                     source.pitch(), source.width(), source.height(), kind);
    assert(err == cudaSuccess);
}

/**
 * @brief Copy a contiguous 3D pitched memory from a host/device memory to a
 * host/device memory using CUDA function cudaMemcpy3D.
 *
 * This function copies the memory wrapped around a HostMemory3D or
 * DeviceMemory3D object to the memory wrapped around a HostMemory3D or
 * DeviceMemory3D. The type of the memory copying is given by cudaMemcpyKind
 * enum which can be one of cudaMemcpyHostToDevice, cudaMemcpyHostToHost,
 * cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice.
 *
 * See cudaMemcpy function documentation to learn more about the memory copying
 * procedure intrinsics.
 *
 * @tparam DstPitchedMemory3D Type of the destination memory. See HostMemory3D
 * and DeviceMemory3D.
 * @tparam SrcPitchedMemory3D Type of the source memory. See HostMemory3D and
 * DeviceMemory3D.
 * @param destination Destination memory object.
 * @param source Source memory object.
 * @param kind Kind of the copying to be performed
 *
 * @throws If the copying procedure is not successful, an assertion error is
 * thrown.
 */
template <typename DstPitchedMemory3D, typename SrcPitchedMemory3D>
void copy3D(DstPitchedMemory3D& destination, const SrcPitchedMemory3D& source,
            cudaMemcpyKind kind) {
    cudaMemcpy3DParms params = {nullptr};
    params.srcPtr = source.pitched_ptr();
    params.dstPtr = destination.pitched_ptr();
    params.extent = source.extent();
    params.kind = kind;

    cudaError_t err = cudaMemcpy3D(&params);
    assert(err == cudaSuccess);
}

} // namespace cuda
} // namespace bnmf_algs
