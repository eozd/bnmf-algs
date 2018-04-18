#include "defs.hpp"
#include <cuda_runtime.h>

namespace bnmf_algs {
namespace cuda {
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

    auto err = cudaMemcpy3D(&params);
    BNMF_ASSERT(err == cudaSuccess, "Error copying memory in cuda::copy3D");
}
} // namespace cuda
} // namespace bnmf_algs
