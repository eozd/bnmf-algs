#pragma once

#include "defs.hpp"
#include <cuda_runtime.h>
#include <type_traits>

namespace bnmf_algs {
namespace cuda {

/**
 * @brief Infer the value of cudaMemcpyKind enum to be used with CUDA copying
 * functions from the types of the memory objects passed to copy1D, copy2D,
 * copy3D.
 *
 * @tparam DstMemory Destination memory type passed to copyXD.
 * @tparam SrcMemory Source memory type passed to copyXD.
 * @tparam HostMemoryBase A template template type representing the base of the
 * host memories that are passed to a copyXD function. For copy1D, this should
 * be HostMemory1D; for copy2D it should be HostMemory2D, and so on.
 * @tparam DeviceMemoryBase A template template type representing the base of
 * the device memories that are passed to a copyXD function. For copy1D, this
 * should be DeviceMemory1D; for copy2D it should be DeviceMemory2D, and so on.
 *
 * @return The inferred value of cudaMemcpyKind enum.
 *
 * @remark If one of host/device to host/device values couldn't be inferred,
 * the function returns cudaMemcpyDefault.
 */
template <typename DstMemory, typename SrcMemory,
          template <typename> class HostMemoryBase,
          template <typename> class DeviceMemoryBase>
constexpr cudaMemcpyKind infer_copy_kind() {
    // Type of the entries in memory objects
    typedef typename DstMemory::value_type DstT;
    typedef typename SrcMemory::value_type SrcT;

    // Type of the memory objects without cv qualifiers
    typedef typename std::remove_cv<DstMemory>::type DstType;
    typedef typename std::remove_cv<SrcMemory>::type SrcType;

    // Infer the cudaMemcpyKind from type of the memory objects
    // We have to write using ternaries due to C++11 restriction in CUDA 8
    return
        // Host <-- Host
        (std::is_same<DstType, HostMemoryBase<DstT>>::value &&
         std::is_same<SrcType, HostMemoryBase<SrcT>>::value)
            ? cudaMemcpyKind::cudaMemcpyHostToHost
            :

            // Host <-- Device
            (std::is_same<DstType, HostMemoryBase<DstT>>::value &&
             std::is_same<SrcType, DeviceMemoryBase<SrcT>>::value)
                ? cudaMemcpyKind::cudaMemcpyDeviceToHost
                :

                // Device <-- Host
                (std::is_same<DstType, DeviceMemoryBase<DstT>>::value &&
                 std::is_same<SrcType, HostMemoryBase<SrcT>>::value)
                    ? cudaMemcpyKind::cudaMemcpyHostToDevice
                    :

                    // Device <-- Device
                    (std::is_same<DstType, DeviceMemoryBase<DstT>>::value &&
                     std::is_same<SrcType, DeviceMemoryBase<SrcT>>::value)
                        ? cudaMemcpyKind::cudaMemcpyDeviceToDevice
                        : cudaMemcpyKind::cudaMemcpyDefault;
}

/**
 * @brief Copy a contiguous 1D memory from a host/device memory to a host/device
 * memory using CUDA function cudaMemcpy.
 *
 * This function copies the memory wrapped around a HostMemory1D or
 * DeviceMemory1D object to the memory wrapped around a HostMemory1D or
 * DeviceMemory1D. The type of the memory copying to be performed is inferred
 * from the types of memory objects at compile-time.
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
 *
 * @throws Static assertion error if one of host/device to host/device enum
 * values could not be inferred
 *
 * @throws Assertion error if the copying procedure is not successful
 */
template <typename DstMemory1D, typename SrcMemory1D>
void copy1D(DstMemory1D& destination, const SrcMemory1D& source) {
    static constexpr cudaMemcpyKind kind =
        infer_copy_kind<DstMemory1D, SrcMemory1D, HostMemory1D,
                        DeviceMemory1D>();
    static_assert(kind != cudaMemcpyDefault,
                  "Invalid copy direction in cuda::copy1D");

    auto err =
        cudaMemcpy(destination.data(), source.data(), source.bytes(), kind);
    BNMF_ASSERT(err == cudaSuccess, "Error copying memory in cuda::copy1D");
}

/**
 * @brief Copy a contiguous 2D pitched memory from a host/device memory to a
 * host/device memory using CUDA function cudaMemcpy2D.
 *
 * This function copies the memory wrapped around a HostMemory2D or
 * DeviceMemory2D object to the memory wrapped around a HostMemory2D or
 * DeviceMemory2D. The type of the memory copying to be performed is inferred
 * from the types of memory objects at compile-time.
 *
 * See cudaMemcpy function documentation to learn more about the memory copying
 * procedure intrinsics.
 *
 * @tparam DstMemory2D Type of the destination memory. See HostMemory2D and
 * DeviceMemory2D.
 * @tparam SrcMemory2D Type of the source memory. See HostMemory2D and
 * DeviceMemory2D.
 * @param destination Destination memory object.
 * @param source Source memory object.
 *
 * @throws Static assertion error if one of host/device to host/device enum
 * values could not be inferred
 *
 * @throws Assertion error if the copying procedure is not successful
 */
template <typename DstPitchedMemory2D, typename SrcPitchedMemory2D>
void copy2D(DstPitchedMemory2D& destination, const SrcPitchedMemory2D& source) {
    static constexpr cudaMemcpyKind kind =
        infer_copy_kind<DstPitchedMemory2D, SrcPitchedMemory2D, HostMemory2D,
                        DeviceMemory2D>();
    static_assert(kind != cudaMemcpyDefault,
                  "Invalid copy direction in cuda::copy2D");

    auto err =
        cudaMemcpy2D(destination.data(), destination.pitch(), source.data(),
                     source.pitch(), source.width(), source.height(), kind);
    BNMF_ASSERT(err == cudaSuccess, "Error copying memory in cuda::copy2D");
}

/**
 * @brief Copy a contiguous 3D pitched memory from a host/device memory to a
 * host/device memory using CUDA function cudaMemcpy3D.
 *
 * This function copies the memory wrapped around a HostMemory3D or
 * DeviceMemory3D object to the memory wrapped around a HostMemory3D or
 * DeviceMemory3D. The type of the memory copying to be performed is inferred
 * from the types of memory objects at compile-time.
 *
 * See cudaMemcpy function documentation to learn more about the memory copying
 * procedure intrinsics.
 *
 * @tparam DstMemory3D Type of the destination memory. See HostMemory3D and
 * DeviceMemory3D.
 * @tparam SrcMemory3D Type of the source memory. See HostMemory3D and
 * DeviceMemory3D.
 * @param destination Destination memory object.
 * @param source Source memory object.
 *
 * @throws Static assertion error if one of host/device to host/device enum
 * values could not be inferred
 *
 * @throws Assertion error if the copying procedure is not successful
 */
template <typename DstPitchedMemory3D, typename SrcPitchedMemory3D>
void copy3D(DstPitchedMemory3D& destination, const SrcPitchedMemory3D& source) {
    static constexpr cudaMemcpyKind kind =
        infer_copy_kind<DstPitchedMemory3D, SrcPitchedMemory3D, HostMemory3D,
                        DeviceMemory3D>();
    static_assert(kind != cudaMemcpyDefault,
                  "Invalid copy direction in cuda::copy3D");

    cudaMemcpy3DParms params = {nullptr};
    params.srcPtr = source.pitched_ptr();
    params.dstPtr = destination.pitched_ptr();
    params.extent = source.extent();
    params.kind = kind;

    auto err = cudaMemcpy3D(&params);
    BNMF_ASSERT(err == cudaSuccess, "Error copying memory in cuda::copy3D");
}
} // namespace cuda
} // namespace bnmf_algs
