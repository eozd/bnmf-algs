#include "defs.hpp"
#include "cuda/tensor_ops.hpp"
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using namespace std::chrono;
using namespace bnmf_algs;

bool close(double a, double b) {
    return std::abs(a - b) <= std::numeric_limits<double>::epsilon();
}

int main() {
    time_point<high_resolution_clock> begin, end;

    begin = high_resolution_clock::now();
    cuda::init();
    end = high_resolution_clock::now();
    std::cout << "Elapsed Init: "
              << duration_cast<milliseconds>(end - begin).count() << " ms"
              << std::endl;

    long x = 1000, y = 4000, z = 25;

    tensord<3> S(x, y, z);
    S.setRandom();

    // Reduction on CPU
    begin = high_resolution_clock::now();
    auto sums = cuda::tensor_sums(S);
    const auto& S_pjk = sums[0];
    const auto& S_ipk = sums[1];
    const auto& S_ijp = sums[2];
    end = high_resolution_clock::now();
    std::cout << "Elapsed CUDA: "
              << duration_cast<milliseconds>(end - begin).count() << " ms"
              << std::endl;

    // Reduction on CPU
    auto num_threads = std::thread::hardware_concurrency();
    Eigen::ThreadPool tp(num_threads);
    Eigen::ThreadPoolDevice dev(&tp, num_threads);

    begin = high_resolution_clock::now();
    tensord<2> E_pjk(y, z);
    tensord<2> E_ipk(x, z);
    tensord<2> E_ijp(x, y);
    E_pjk.device(dev) = S.sum(shape<1>({0}));
    E_ipk.device(dev) = S.sum(shape<1>({1}));
    E_ijp.device(dev) = S.sum(shape<1>({2}));
    end = high_resolution_clock::now();
    std::cout << "Elapsed CPU: "
              << duration_cast<milliseconds>(end - begin).count() << " ms"
              << std::endl;

    // Compare CPU and GPU results
    for (long i = 0; i < x; ++i) {
        for (long j = 0; j < y; ++j) {
            assert(close(S_ijp(i, j), E_ijp(i, j)));
        }
    }

    for (long i = 0; i < x; ++i) {
        for (long k = 0; k < z; ++k) {
            assert(close(S_ipk(i, k), E_ipk(i, k)));
        }
    }

    for (long j = 0; j < y; ++j) {
        for (long k = 0; k < z; ++k) {
            assert(close(S_pjk(j, k), E_pjk(j, k)));
        }
    }

    std::cout << "Done" << std::endl;
    return 0;
}
