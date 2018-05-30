#pragma once


#include <celero/Celero.h>

#include "bench_utils.hpp"
#include "bnmf_algs.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

namespace swimmer_bench_vars {
    size_t x = 1024;
    size_t y = 256;

    bnmf_algs::matrixd read_mat(std::istream& input) {
        std::vector<double> data;
        size_t n_rows = 0;
        std::string line;
        double value;
        while (std::getline(input, line)) {
            size_t index = 0, prev_index = 0;
            index = line.find(' ');
            while (prev_index != std::string::npos) {
                value =
                        std::atof(line.substr(prev_index, index - prev_index).data());
                data.push_back(value);

                prev_index = index;
                index = line.find(' ', index + 1);
            }
            ++n_rows;
        }
        auto n_cols = data.size() / n_rows;

        bnmf_algs::matrixd X(n_rows, n_cols);
        for (size_t i = 0; i < n_rows; ++i) {
            for (size_t j = 0; j < n_cols; ++j) {
                X(i, j) = data[i*n_cols + j];
            }
        }

        return X;
    }

    const bnmf_algs::matrixd& swimmer_mat() {
        static std::ifstream file("swimmer.txt");
        static bnmf_algs::matrixd X = read_mat(file);

        return X;
    }

    size_t r = 17;
} // namespace swimmer_bench_vars

/**************************** Original bld_mult *******************************/

BASELINE(swimmer, bld_mult, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace swimmer_bench_vars;

    const auto& X = swimmer_mat();
    auto params = make_params<double>(x, y, r);
    tensord<3> S;
    celero::DoNotOptimizeAway(S = bld::bld_mult(X, r, params, 250, true));
}

BENCHMARK(swimmer, bld_mult_cuda, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace swimmer_bench_vars;

    const auto& X = swimmer_mat();
    auto params = make_params<double>(x, y, r);
    tensord<3> S;
    celero::DoNotOptimizeAway(S = bld::bld_mult_cuda(X, r, params, 250, true));
}

BENCHMARK(swimmer, bld_add, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace swimmer_bench_vars;

    const auto& X = swimmer_mat();
    auto params = make_params<double>(x, y, r);
    tensord<3> S;
    celero::DoNotOptimizeAway(S = bld::bld_add(X, r, params, 250, true));
}

BENCHMARK(swimmer, bld_appr, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace swimmer_bench_vars;

    const auto& X = swimmer_mat();
    auto params = make_params<double>(x, y, r);
    tensord<3> S;
    matrixd _;
    celero::DoNotOptimizeAway(std::tie(S, _, _) = bld::bld_appr(X, r, params, 250, true));
}

BENCHMARK(swimmer, seq_greedy_bld, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace swimmer_bench_vars;

    const auto& X = swimmer_mat();
    auto params = make_params<double>(x, y, r);
    tensord<3> S;
    celero::DoNotOptimizeAway(S = bld::seq_greedy_bld(X, r, params));
}

BENCHMARK(swimmer, collapsed_gibbs, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace swimmer_bench_vars;

    const auto& X = swimmer_mat();
    auto params = make_params<double>(x, y, r);
    auto gen = bld::collapsed_gibbs(X, r, params, 250);
    for (const auto& _ : gen);

    const auto S = *(gen.begin());
}

BENCHMARK(swimmer, collapsed_icm, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace swimmer_bench_vars;

    const auto& X = swimmer_mat();
    auto params = make_params<double>(x, y, r);
    tensord<3> S;
    celero::DoNotOptimizeAway(S = bld::collapsed_icm(X, r, params, 250));
}

BENCHMARK(swimmer, online_EM, 1, 1) {
    using namespace bnmf_algs;
    using namespace benchmark;
    using namespace swimmer_bench_vars;

    const auto& X = swimmer_mat();
    shape<3> tensor_shape{x, y, r};
    auto param_vec = alloc_model::make_EM_params<double>(tensor_shape);
    bld::EMResult<double> em_res;
    celero::DoNotOptimizeAway(em_res = bld::online_EM(X, param_vec, 250, true));
}
