#include "../catch2.hpp"
#include "cuda/memory.hpp"
#include "cuda/tensor_ops.hpp"
#include "cuda/util.hpp"
#include "defs.hpp"
#include "util/util.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono;
using namespace bnmf_algs;

TEST_CASE("Test tensor_sums", "[tensor_ops]") {
    SECTION("Same results on GPU and CPU") {
        cuda::init(0);

        size_t x = 1024, y = 456, z = 17;

        tensord<3> S(x, y, z);
        S.setRandom();

        tensord<2> S_pjk(y, z);
        tensord<2> S_ipk(x, z);
        tensord<2> S_ijp(x, y);

        // Reduction on GPU
        {
            cuda::HostMemory3D<const double> S_host(S.data(), x, y, z);
            std::array<cuda::HostMemory2D<double>, 3> result_arr = {
                cuda::HostMemory2D<double>(S_pjk.data(), y, z),
                cuda::HostMemory2D<double>(S_ipk.data(), x, z),
                cuda::HostMemory2D<double>(S_ijp.data(), x, y)};
            // allocate GPU memory
            cuda::DeviceMemory3D<double> S_device(x, y, z);
            std::array<cuda::DeviceMemory2D<double>, 3> device_result_arr = {
                cuda::DeviceMemory2D<double>(y, z),
                cuda::DeviceMemory2D<double>(x, z),
                cuda::DeviceMemory2D<double>(x, y)};

            // copy S to GPU
            cuda::copy3D(S_device, S_host);

            // calculate sums
            cuda::tensor_sums(S_device, device_result_arr);

            // copy from GPU to main memory
            for (size_t i = 0; i < 3; ++i) {
                cuda::copy2D(result_arr[i], device_result_arr[i]);
            }
        }

        tensord<2> E_pjk = S.sum(shape<1>({0}));
        tensord<2> E_ipk = S.sum(shape<1>({1}));
        tensord<2> E_ijp = S.sum(shape<1>({2}));

        Eigen::Map<const matrixd> S_pjk_mat(S_pjk.data(), y, z);
        Eigen::Map<const matrixd> E_pjk_mat(E_pjk.data(), y, z);

        Eigen::Map<const matrixd> S_ipk_mat(S_ipk.data(), x, z);
        Eigen::Map<const matrixd> E_ipk_mat(E_ipk.data(), x, z);

        Eigen::Map<const matrixd> S_ijp_mat(S_ijp.data(), x, y);
        Eigen::Map<const matrixd> E_ijp_mat(E_ijp.data(), x, y);

        REQUIRE(S_pjk_mat.isApprox(E_pjk_mat));
        REQUIRE(S_ipk_mat.isApprox(E_ipk_mat));
        REQUIRE(S_ijp_mat.isApprox(E_ijp_mat));
    }
}

