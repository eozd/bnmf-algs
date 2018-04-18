#include "defs.hpp"
#include <cstddef>

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
        : m_dims(shape<1>{num_elems}), m_data(data){};

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
    size_t bytes() const { return m_dims[0] * sizeof(T); }

    /**
     * @brief Get the dimensions of this memory region in terms of elements.
     *
     * @return A bnmf_algs::shape representing the dimension.
     */
    shape<1> dims() const { return m_dims; }

  private:
    /**
     * @brief Dimension (length) of this memory region.
     */
    shape<1> m_dims;
    /**
     * @brief Beginning address of the memory sequence.
     */
    T* m_data;
};
} // namespace cuda
} // namespace bnmf_algs
