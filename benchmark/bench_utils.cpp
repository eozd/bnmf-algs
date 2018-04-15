#include "bench_utils.hpp"

using namespace bnmf_algs;

matrixd benchmark::make_matrix(long x, long y, double beg, double scale) {
    matrixd res = matrixd::Random(x, y) + matrixd::Constant(x, y, beg + 1);
    res = res.array() * (scale / 2);

    return res;
}

allocation_model::AllocModelParams benchmark::make_params(long x, long y,
                                                          long z) {
    allocation_model::AllocModelParams params(
        100, 1, std::vector<double>(x, 0.05), std::vector<double>(z, 10));
    params.beta.back() = 60;

    return params;
}
