#include "alloc_model/alloc_model_params.hpp"

using namespace bnmf_algs;

alloc_model::AllocModelParams::AllocModelParams(
    double a, double b, const std::vector<double>& alpha,
    const std::vector<double>& beta)
    : a(a), b(b), alpha(alpha), beta(beta) {}

alloc_model::AllocModelParams::AllocModelParams(
    const shape<3>& tensor_shape)
    : a(1.0), b(10.0), alpha(tensor_shape[0], 1.0), beta(tensor_shape[2], 1.0) {
}
