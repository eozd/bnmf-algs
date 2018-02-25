#include "decomposition.hpp"
#include "wrappers.hpp"
#include <algorithm>

bnmf_algs::tensor3d_t
bnmf_algs::seq_greedy_bld(const matrix_t& X, size_t z,
                          const std::vector<double>& alpha,
                          const std::vector<double>& beta) {
    // todo: nonnegative X
    // todo: size(beta) == z
    // todo: size(alpha) == x
    long x = X.rows();
    long y = X.cols();
    double sig_alpha = std::accumulate(alpha.begin(), alpha.end(), 0.0);

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
        auto rand_discrete = make_gsl_ran_discrete(nonzero_indices.size(), nonzero_values.data());

        size_t idx = gsl_ran_discrete(rand_gen.get(), rand_discrete.get());
        nonzero_values[idx] = std::max(nonzero_values[idx] - 1, 0.0);

        std::tie(ii, jj) = nonzero_indices[idx];
        ll.setZero();

        for (int k = 0; k < z; ++k) {
            ll[k] = std::log(alpha[ii] + S_ipk(ii,k)) - std::log(sig_alpha + S_ppk(k));
            ll[k] += -(std::log(1 + S(ii, jj, k)) - std::log(beta[k] + S_pjk(jj, k)));
        }

        auto ll_begin = ll.data();
        auto kmax = std::distance(ll_begin, std::max_element(ll_begin, ll_begin + z));

        S(ii, jj, kmax) += 1;
        S_ipk(ii, kmax) += 1;
        S_ppk(kmax) += 1;
        S_pjk(jj, kmax) += 1;
    }

    return S;
}
