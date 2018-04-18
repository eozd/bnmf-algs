#include <cstddef>

namespace bnmf_algs {
namespace cuda {
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
} // namespace cuda
} // namespace bnmf_algs
