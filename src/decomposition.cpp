#include "decomposition.hpp"
#include "wrappers.hpp"
#include <algorithm>

/**
 * @brief Do parameter checks on seq_greedy_bld parameters and return error
 * message if there is any.
 *
 * @return Error message. If there isn't any error, returns "".
 */
static std::string
seq_greedy_bld_param_checks(const bnmf_algs::matrix_t& X, size_t z,
                            const bnmf_algs::AllocModelParams& model_params) {
    if ((X.array() < 0).any()) {
        return "X must be nonnegative";
    }
    if (model_params.alpha.size() != X.rows()) {
        return "Number of alpha parameters must be equal to number of rows of "
               "X";
    }
    if (model_params.beta.size() != z) {
        return "Number of beta parameters must be equal to z";
    }
    return "";
}

bnmf_algs::tensor3d_t
bnmf_algs::seq_greedy_bld(const matrix_t& X, size_t z,
                          const AllocModelParams& model_params) {
    {
        auto error_msg = seq_greedy_bld_param_checks(X, z, model_params);
        if (error_msg != "") {
            throw std::invalid_argument(error_msg);
        }
    }
    long x = X.rows();
    long y = X.cols();
    double sig_alpha = std::accumulate(model_params.alpha.begin(),
                                       model_params.alpha.end(), 0.0);

    double sum = 0;
    std::vector<std::pair<int, int>> nonzero_indices;
    std::vector<double> nonzero_values;
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            double val = X(i, j);
            if (val > 0) {
                sum += val;
                nonzero_indices.emplace_back(i, j);
                nonzero_values.push_back(val);
            }
        }
    }

    tensor3d_t S(x, y, z);
    S.setZero();
    matrix_t S_ipk = matrix_t::Zero(x, z); // S_{i+k}
    vector_t S_ppk = vector_t::Zero(z);    // S_{++k}
    matrix_t S_pjk = matrix_t::Zero(y, z); // S_{+jk}

    auto rand_gen = make_gsl_rng(gsl_rng_taus);
    int ii, jj;
    vector_t ll(z);
    for (int i = 0; i < (int)sum; ++i) {
        auto rand_discrete = make_gsl_ran_discrete(nonzero_indices.size(),
                                                   nonzero_values.data());

        size_t idx = gsl_ran_discrete(rand_gen.get(), rand_discrete.get());
        nonzero_values[idx] = std::max(nonzero_values[idx] - 1, 0.0);

        std::tie(ii, jj) = nonzero_indices[idx];
        ll.setZero();

        for (int k = 0; k < z; ++k) {
            ll[k] = std::log(model_params.alpha[ii] + S_ipk(ii, k)) -
                    std::log(sig_alpha + S_ppk(k));
            ll[k] += -(std::log(1 + S(ii, jj, k)) -
                       std::log(model_params.beta[k] + S_pjk(jj, k)));
        }

        auto ll_begin = ll.data();
        auto kmax =
            std::distance(ll_begin, std::max_element(ll_begin, ll_begin + z));

        S(ii, jj, kmax) += 1;
        S_ipk(ii, kmax) += 1;
        S_ppk(kmax) += 1;
        S_pjk(jj, kmax) += 1;
    }

    return S;
}
