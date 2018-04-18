#include "defs.hpp"
#include <cstddef>
#include <cuda_runtime.h>

namespace bnmf_algs {
namespace cuda {
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
        : m_dims(shape<3>{first_dim, second_dim, third_dim}),
          m_extent(
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

    /**
     * @brief Get the dimensions of this memory region in terms of elements.
     *
     * @return A bnmf_algs::shape object of the form {first_dim, second_dim,
     * third_dim}.
     */
    shape<3> dims() const { return m_dims; }

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
