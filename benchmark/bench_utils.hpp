#pragma once

#include "bnmf_algs.hpp"

namespace benchmark {
bnmf_algs::matrix_t make_matrix(long x, long y, double beg, double scale);
bnmf_algs::allocation_model::AllocModelParams make_params(long x, long y, long z);
} // namespace benchmark
