#pragma once
#include "defs.hpp"
#include <cuda_runtime.h>

namespace bnmf_algs {
namespace cuda {
/**
 * @brief Initialize CUDA runtime.
 *
 * This function initializes CUDA runtime so that future CUDA library calls
 * don't incur the cost of initializing the library.
 *
 * @param device ID of the GPU device to set.
 */
template <typename Integer> void init(Integer device) {
    auto err = cudaSetDevice(device);
    BNMF_ASSERT(err == cudaSuccess, "Error setting CUDA device in cuda::init");
}

/**
 * @brief Return ceiling of integer division between given parameters.
 *
 * This function returns \f$\ceil{\frac{a}{b}}\f$ for parameters a and b.
 *
 * @tparam Integer An integer type such as int, long, size_t, ...
 * @param a Nominator.
 * @param b Denominator.
 * @return Ceiling of \f$\frac{a}{b}\f$ as an integer.
 */
template <typename Integer> Integer idiv_ceil(Integer a, Integer b) {
    return a / b + (a % b != 0);
}
} // namespace cuda
} // namespace bnmf_algs
