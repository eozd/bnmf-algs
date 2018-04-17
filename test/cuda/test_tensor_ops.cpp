#include "../catch2.hpp"
#include "cuda/tensor_ops.hpp"
#include "cuda/util.hpp"
#include "defs.hpp"
#include "util/util.hpp"
#include <iostream>

using namespace bnmf_algs;

TEST_CASE("Test update_grad_plus", "[tensor_ops]") {
    SECTION("Same results on GPU and CPU") {
        cuda::init(0);

        size_t x = 131, y = 456, z = 89;

        matrixd data =
            matrixd::Random(x, y * z) + matrixd::Constant(x, y * z, 1);
        tensord<3> S(x, y, z);
        std::copy(data.data(), data.data() + x * y * z, S.data());

        matrixd beta_eph = matrixd::Random(y, z) + matrixd::Constant(y, z, 1);

        tensord<3> actual(x, y, z);

        // Reduction on GPU
        cuda::bld_mult::update_grad_plus(S, beta_eph, actual);

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
        cuda::apply_psi(actual.data(), x * y * z);

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

        size_t x = 131, y = 456, z = 89;

        tensord<3> S(x, y, z);
        S.setRandom();

        // Reduction on GPU
        auto sums = cuda::tensor_sums(S);
        const auto& S_pjk = sums[0];
        const auto& S_ipk = sums[1];
        const auto& S_ijp = sums[2];

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