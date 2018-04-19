#include "../../catch2.hpp"
#include "bld/bld_mult/bld_mult_cuda_funcs.hpp"
#include "cuda/memory.hpp"
#include "cuda/util.hpp"
#include "defs.hpp"
#include "util/util.hpp"
#include <chrono>
#include <iostream>

using namespace bnmf_algs;

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

            details::bld_mult_update_grad_plus(S_device, beta_eph_device,
                                               actual_device);

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

            details::bld_mult_update_nom(X_reciprocal_device, grad_minus_device,
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

TEST_CASE("Test update_denom", "[tensor_ops]") {
    SECTION("Same results on GPU and CPU") {
        const double eps = 1e-50;
        cuda::init(0);

        size_t x = 356, y = 478, z = 17;

        // initialization
        matrixd X = matrixd::Random(x, y) + matrixd::Constant(x, y, 10);
        matrixd X_reciprocal = matrixd::Constant(x, y, 1).array() / X.array();
        tensord<3> S(x, y, z);
        {
            matrixd data =
                matrixd::Random(x, y * z) + matrixd::Constant(x, y * z, 1);
            std::copy(data.data(), data.data() + x * y * z, S.data());
        }
        tensord<3> grad_plus(S);

        matrixd actual(x, y);
        // Calculate on GPU
        {
            cuda::HostMemory2D<double> X_reciprocal_host(X_reciprocal.data(), x,
                                                         y);
            cuda::HostMemory3D<double> grad_plus_host(grad_plus.data(), x, y,
                                                      z);
            cuda::HostMemory3D<double> S_host(S.data(), x, y, z);
            cuda::HostMemory2D<double> actual_host(actual.data(), x, y);

            cuda::DeviceMemory2D<double> X_reciprocal_device(x, y);
            cuda::DeviceMemory3D<double> grad_plus_device(x, y, z);
            cuda::DeviceMemory3D<double> S_device(x, y, z);
            cuda::DeviceMemory2D<double> actual_device(x, y);

            cuda::copy2D(X_reciprocal_device, X_reciprocal_host);
            cuda::copy3D(grad_plus_device, grad_plus_host);
            cuda::copy3D(S_device, S_host);

            details::bld_mult_update_denom(
                X_reciprocal_device, grad_plus_device, S_device, actual_device);
            cuda::copy2D(actual_host, actual_device);
        }

        // Calculate on CPU
        matrixd expected(x, y);
        for (size_t i = 0; i < x; ++i) {
            for (size_t j = 0; j < y; ++j) {
                double xdiv = 1.0 / (X(i, j) + eps);
                double denom_sum = 0;
                for (size_t k = 0; k < z; ++k) {
                    denom_sum += (S(i, j, k) * xdiv * grad_plus(i, j, k));
                }
                expected(i, j) = denom_sum;
            }
        }

        REQUIRE(actual.isApprox(expected));
    }
}

TEST_CASE("Test update_S", "[tensor_ops]") {
    SECTION("Same results on GPU and CPU") {
        const double eps = 1e-50;
        cuda::init(0);

        size_t x = 356, y = 478, z = 17;

        // initialization
        matrixd X = matrixd::Random(x, y) + matrixd::Constant(x, y, 10);
        matrixd nom_mult = matrixd::Random(x, y) + matrixd::Constant(x, y, 3);
        matrixd denom_mult = matrixd::Random(x, y) + matrixd::Constant(x, y, 5);
        matrixd grad_minus = matrixd::Random(x, z) + matrixd::Constant(x, z, 2);
        tensord<3> S(x, y, z);
        {
            matrixd data =
                matrixd::Random(x, y * z) + matrixd::Constant(x, y * z, 1);
            std::copy(data.data(), data.data() + x * y * z, S.data());
        }
        tensord<3> expected(S);
        tensord<3> grad_plus(S);
        tensord<2> S_ijp = S.sum(shape<1>({2}));

        // calculate on GPU
        {
            cuda::HostMemory2D<double> X_host(X.data(), x, y);
            cuda::HostMemory2D<double> nom_host(nom_mult.data(), x, y);
            cuda::HostMemory2D<double> denom_host(denom_mult.data(), x, y);
            cuda::HostMemory2D<double> grad_minus_host(grad_minus.data(), x, z);
            cuda::HostMemory3D<double> grad_plus_host(grad_plus.data(), x, y,
                                                      z);
            cuda::HostMemory3D<double> S_host(S.data(), x, y, z);
            cuda::HostMemory2D<double> S_ijp_host(S_ijp.data(), x, y);

            cuda::DeviceMemory2D<double> X_device(x, y);
            cuda::DeviceMemory2D<double> nom_device(x, y);
            cuda::DeviceMemory2D<double> denom_device(x, y);
            cuda::DeviceMemory2D<double> grad_minus_device(x, z);
            cuda::DeviceMemory3D<double> grad_plus_device(x, y, z);
            cuda::DeviceMemory3D<double> S_device(x, y, z);
            cuda::DeviceMemory2D<double> S_ijp_device(x, y);

            cuda::copy2D(X_device, X_host);
            cuda::copy2D(nom_device, nom_host);
            cuda::copy2D(denom_device, denom_host);
            cuda::copy2D(grad_minus_device, grad_minus_host);
            cuda::copy3D(grad_plus_device, grad_plus_host);
            cuda::copy3D(S_device, S_host);
            cuda::copy2D(S_ijp_device, S_ijp_host);

            details::bld_mult_update_S(X_device, nom_device, denom_device,
                                       grad_minus_device, grad_plus_device,
                                       S_ijp_device, S_device);
            cuda::copy3D(S_host, S_device);
        }

        // calculate on CPU
        for (size_t i = 0; i < x; ++i) {
            for (size_t j = 0; j < y; ++j) {
                double s_ij = S_ijp(i, j);
                double x_over_s = X(i, j) / (s_ij + eps);
                for (size_t k = 0; k < z; ++k) {
                    expected(i, j, k) *=
                        (grad_plus(i, j, k) + nom_mult(i, j)) * x_over_s /
                        (grad_minus(i, k) + denom_mult(i, j) + eps);
                }
            }
        }

        Eigen::Map<matrixd> actual_mat(S.data(), x, y * z);
        Eigen::Map<matrixd> expect_mat(expected.data(), x, y * z);

        REQUIRE(actual_mat.isApprox(expect_mat));
    }
}
