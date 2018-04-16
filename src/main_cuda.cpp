#include "cuda/tensor_ops.hpp"
#include "defs.hpp"
#include <bnmf_algs.hpp>
#include <cassert>
#include <chrono>
#include <cmath>
#include <gsl/gsl_sf_psi.h>
#include <iostream>
#include <limits>
#include <vector>

using namespace std::chrono;
using namespace bnmf_algs;

bool close(double a, double b, double eps) { return std::abs(a - b) <= eps; }

//int main() {
//    time_point<high_resolution_clock> begin, end;
//
//    begin = high_resolution_clock::now();
//    cuda::init();
//    end = high_resolution_clock::now();
//    std::cout << "Elapsed Init: "
//              << duration_cast<milliseconds>(end - begin).count() << " ms"
//              << std::endl;
//
//    long x = 1000, y = 4000, z = 25;
//
//    tensord<3> S(x, y, z);
//    S.setRandom();
//
//    // Reduction on GPU
//    begin = high_resolution_clock::now();
//    auto sums = cuda::tensor_sums(S);
//    const auto& S_pjk = sums[0];
//    const auto& S_ipk = sums[1];
//    const auto& S_ijp = sums[2];
//    end = high_resolution_clock::now();
//    std::cout << "Elapsed CUDA: "
//              << duration_cast<milliseconds>(end - begin).count() << " ms"
//              << std::endl;
//
//    // Reduction on CPU
//    auto num_threads = std::thread::hardware_concurrency();
//    Eigen::ThreadPool tp(num_threads);
//    Eigen::ThreadPoolDevice dev(&tp, num_threads);
//
//    begin = high_resolution_clock::now();
//    tensord<2> E_pjk(y, z);
//    tensord<2> E_ipk(x, z);
//    tensord<2> E_ijp(x, y);
//    E_pjk.device(dev) = S.sum(shape<1>({0}));
//    E_ipk.device(dev) = S.sum(shape<1>({1}));
//    E_ijp.device(dev) = S.sum(shape<1>({2}));
//    end = high_resolution_clock::now();
//    std::cout << "Elapsed CPU: "
//              << duration_cast<milliseconds>(end - begin).count() << " ms"
//              << std::endl;
//
//    // Compare CPU and GPU results
//    for (long i = 0; i < x; ++i) {
//        for (long j = 0; j < y; ++j) {
//            assert(close(S_ijp(i, j), E_ijp(i, j), 1e-13));
//        }
//    }
//
//    for (long i = 0; i < x; ++i) {
//        for (long k = 0; k < z; ++k) {
//            assert(close(S_ipk(i, k), E_ipk(i, k), 1e-13));
//        }
//    }
//
//    for (long j = 0; j < y; ++j) {
//        for (long k = 0; k < z; ++k) {
//            assert(close(S_pjk(j, k), E_pjk(j, k), 1e-13));
//        }
//    }
//
//    std::cout << "Done" << std::endl;
//    return 0;
//}

// int main() {
//    time_point<high_resolution_clock> begin, end;
//
//    begin = high_resolution_clock::now();
//    cuda::init();
//    end = high_resolution_clock::now();
//    std::cout << "Elapsed Init: "
//              << duration_cast<milliseconds>(end - begin).count() << " ms"
//              << std::endl;
//
//    size_t x = 1024, y = 400, z = 17;
//
//    matrixd data = matrixd::Random(x, y*z) + matrixd::Constant(x, y*z, 1);
//
//    Eigen::TensorMap<tensord<3>> S(data.data(), x, y, z);
//
//    // Reduction on GPU
//    begin = high_resolution_clock::now();
//    cuda::apply_psi(S.data(), x * y * z);
//    end = high_resolution_clock::now();
//    std::cout << "Elapsed CUDA: "
//              << duration_cast<milliseconds>(end - begin).count() << " ms"
//              << std::endl;
//
//    // Reduction on CPU
//    begin = high_resolution_clock::now();
//    #pragma omp parallel for schedule(static)
//    for (size_t i = 0; i < x; ++i) {
//        for (size_t j = 0; j < y; ++j) {
//            for (size_t k = 0; k < z; ++k) {
//                S(i, j, k) = util::psi_appr(S(i, j, k));
//            }
//        }
//    }
//    end = high_resolution_clock::now();
//    std::cout << "Elapsed CPU: "
//              << duration_cast<milliseconds>(end - begin).count() << " ms"
//              << std::endl;
//
//    std::cout << "Done" << std::endl;
//    return 0;
//}

 int main() {
    time_point<high_resolution_clock> begin, end;

    begin = high_resolution_clock::now();
    cuda::init();
    end = high_resolution_clock::now();
    std::cout << "Elapsed Init: "
              << duration_cast<milliseconds>(end - begin).count() << " ms"
              << std::endl;

    size_t x = 1000, y = 1000, z = 25;

    matrixd data =
        matrixd::Random(x, y * z) + matrixd::Constant(x, y * z, 1);
    Eigen::TensorMap<tensord<3>> S(data.data(), x, y, z);

    matrixd beta_eph = matrixd::Random(y, z) + matrixd::Constant(y, z, 1);

    tensord<3> grad_plus(x, y, z);

    // Reduction on GPU
    begin = high_resolution_clock::now();
    cuda::bld_mult::update_grad_plus(S, beta_eph, grad_plus);
    end = high_resolution_clock::now();
    std::cout << "Elapsed CUDA: "
              << duration_cast<milliseconds>(end - begin).count() << " ms"
              << std::endl;

    tensord<3> grad_plus_cpu(x, y, z);
    // Reduction on CPU
    begin = high_resolution_clock::now();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            for (int k = 0; k < z; ++k) {
                grad_plus_cpu(i, j, k) = util::psi_appr(beta_eph(j, k)) -
                                         util::psi_appr(S(i, j, k) + 1);
            }
        }
    }
    end = high_resolution_clock::now();
    std::cout << "Elapsed CPU: "
              << duration_cast<milliseconds>(end - begin).count() << " ms"
              << std::endl;

    // compare results
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            for (int k = 0; k < z; ++k) {
                assert(
                    close(grad_plus(i, j, k), grad_plus_cpu(i, j, k), 1e-10));
            }
        }
    }

    std::cout << "Done" << std::endl;
    return 0;
}
