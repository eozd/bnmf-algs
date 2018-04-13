#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>
#include <cassert>
#include "defs.hpp"

using namespace std::chrono;
using namespace bnmf_algs;

extern matrix_t cuda_tensor_sum(const tensord<3>&, size_t);
extern void cuda_init();

bool close(double a, double b) {
    return std::abs(a - b) <= std::numeric_limits<double>::epsilon();
}

int main() {
    time_point<high_resolution_clock> begin, end;

    begin = high_resolution_clock::now();
    cuda_init();
    end = high_resolution_clock::now();
    std::cout << "Elapsed Init: " << duration_cast<milliseconds>(end - begin).count() << " ms" << std::endl;

    long x = 2000, y = 800, z = 50;

    tensord<3> S(x, y, z);
    S.setRandom();

    begin = high_resolution_clock::now();
    matrix_t S_pjk = cuda_tensor_sum(S, 0);
    matrix_t S_ipk = cuda_tensor_sum(S, 1);
    matrix_t S_ijp = cuda_tensor_sum(S, 2);
    end = high_resolution_clock::now();
    std::cout << "Elapsed CUDA: " << duration_cast<milliseconds>(end - begin).count() << " ms" << std::endl;

    begin = high_resolution_clock::now();
    tensord<2> expected_first = S.sum(shape<1>({0}));
    tensord<2> expected_second = S.sum(shape<1>({1}));
    tensord<2> expected_third = S.sum(shape<1>({2}));
    end = high_resolution_clock::now();
    std::cout << "Elapsed CPU: " << duration_cast<milliseconds>(end - begin).count() << " ms" << std::endl;

    //for (long i = 0; i < x; ++i) {
    //    for (long j = 0; j < y; ++j) {
    //        assert(close(S_ijp(i, j), expected_third(i, j)));
    //    }
    //}

    //for (long i = 0; i < x; ++i) {
    //    for (long k = 0; k < z; ++k) {
    //        assert(close(S_ipk(i, k), expected_second(i, k)));
    //    }
    //}

    //for (long j = 0; j < y; ++j) {
    //    for (long k = 0; k < z; ++k) {
    //        assert(close(S_pjk(j, k), expected_first(j, k)));
    //    }
    //}

    std::cout << "Done" << std::endl;
    return 0;
}


