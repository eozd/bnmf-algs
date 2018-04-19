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

TEST_CASE("Test apply_psi", "[tensor_ops]") {
    SECTION("Same results on GPU and CPU") {
        cuda::init(0);

        size_t x = 131, y = 456, z = 89;

        matrixd data =
            matrixd::Random(x, y * z) + matrixd::Constant(x, y * z, 1);
        Eigen::TensorMap<tensord<3>> data_tensor(data.data(), x, y, z);

        matrixd data_copy = data;

        // Reduction on GPU
        Eigen::TensorMap<tensord<3>> actual(data_copy.data(), x, y, z);
        cuda::HostMemory1D<double> actual_host(actual.data(), x * y * z);
        cuda::DeviceMemory1D<double> actual_device(x * y * z);
        cuda::copy1D(actual_device, actual_host);
        cuda::apply_psi(actual_device);
        cuda::copy1D(actual_host, actual_device);

        // Reduction on CPU
        tensord<3> expected(x, y, z);
        for (size_t i = 0; i < x; ++i) {
            for (size_t j = 0; j < y; ++j) {
                for (size_t k = 0; k < z; ++k) {
                    expected(i, j, k) = util::psi_appr(data_tensor(i, j, k));
                }
            }
        }

        Eigen::Map<matrixd> actual_mat(actual.data(), x, y * z);
        Eigen::Map<matrixd> expect_mat(expected.data(), x, y * z);

        REQUIRE(actual_mat.isApprox(expect_mat, 1e-10));
    }
}

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

TEST_CASE("Test update_grad_plus", "[tensor_ops]") {
    SECTION("Same results on GPU and CPU") {
        cuda::init(0);

        size_t x = 131, y = 456, z = 89;

        tensord<3> S(x, y, z);
        {
            matrixd data =
                matrixd::Random(x, y * z) + matrixd::Constant(x, y * z, 1);
            std::copy(data.data(), data.data() + x * y * z, S.data());
        }
        matrixd beta_eph = matrixd::Random(y, z) + matrixd::Constant(y, z, 1);
        tensord<3> actual(x, y, z);

        {
            cuda::HostMemory3D<double> S_host(S.data(), x, y, z);
            cuda::HostMemory2D<double> beta_eph_host(beta_eph.data(), y, z);
            cuda::HostMemory3D<double> actual_host(actual.data(), x, y, z);

            cuda::DeviceMemory3D<double> S_device(x, y, z);
            cuda::DeviceMemory2D<double> beta_eph_device(y, z);
            cuda::DeviceMemory3D<double> actual_device(x, y, z);

            cuda::copy3D(S_device, S_host);
            cuda::copy2D(beta_eph_device, beta_eph_host);
            cuda::copy3D(actual_device, actual_host);

            cuda::bld_mult::update_grad_plus(S_device, beta_eph_device,
                                             actual_device);

            cuda::copy3D(S_host, S_device);
            cuda::copy2D(beta_eph_host, beta_eph_device);
            cuda::copy3D(actual_host, actual_device);
        }

        tensord<3> expected(x, y, z);
        // Reduction on CPU
        for (size_t i = 0; i < x; ++i) {
            for (size_t j = 0; j < y; ++j) {
                for (size_t k = 0; k < z; ++k) {
                    expected(i, j, k) = util::psi_appr(beta_eph(j, k)) -
                                        util::psi_appr(S(i, j, k) + 1);
                }
            }
        }

        Eigen::Map<matrixd> actual_mat(actual.data(), x, y * z);
        Eigen::Map<matrixd> expect_mat(expected.data(), x, y * z);

        REQUIRE(actual_mat.isApprox(expect_mat, 1e-12));
    }
}

TEST_CASE("Test update_nom", "[tensor_ops]") {
    SECTION("Same results on GPU and CPU") {
        const double eps = 1e-50;
        cuda::init(0);

        size_t x = 356, y = 478, z = 17;

        // initialization
        matrixd X = matrixd::Random(x, y) + matrixd::Constant(x, y, 10);
        matrixd X_reciprocal = matrixd::Constant(x, y, 1).array() / X.array();
        matrixd grad_minus = matrixd::Random(x, z) + matrixd::Constant(x, z, 2);
        tensord<3> S(x, y, z);
        {
            matrixd data =
                matrixd::Random(x, y * z) + matrixd::Constant(x, y * z, 1);
            std::copy(data.data(), data.data() + x * y * z, S.data());
        }

        matrixd actual(x, y);
        // Calculate on GPU
        {
            cuda::HostMemory2D<double> X_reciprocal_host(X_reciprocal.data(), x,
                                                         y);
            cuda::HostMemory2D<double> grad_minus_host(grad_minus.data(), x, z);
            cuda::HostMemory3D<double> S_host(S.data(), x, y, z);
            cuda::HostMemory2D<double> actual_host(actual.data(), x, y);

            cuda::DeviceMemory2D<double> X_reciprocal_device(x, y);
            cuda::DeviceMemory2D<double> grad_minus_device(x, z);
            cuda::DeviceMemory3D<double> S_device(x, y, z);
            cuda::DeviceMemory2D<double> actual_device(x, y);

            cuda::copy2D(X_reciprocal_device, X_reciprocal_host);
            cuda::copy2D(grad_minus_device, grad_minus_host);
            cuda::copy3D(S_device, S_host);

            cuda::bld_mult::update_nom(X_reciprocal_device, grad_minus_device,
                                       S_device, actual_device);
            cuda::copy2D(actual_host, actual_device);
        }

        // Calculate on CPU
        matrixd expected(x, y);
        for (size_t i = 0; i < x; ++i) {
            for (size_t j = 0; j < y; ++j) {
                double xdiv = 1.0 / (X(i, j) + eps);
                double nom_sum = 0;
                for (size_t k = 0; k < z; ++k) {
                    nom_sum += (S(i, j, k) * xdiv * grad_minus(i, k));
                }
                expected(i, j) = nom_sum;
            }
        }

        REQUIRE(actual.isApprox(expected));
    }
}
