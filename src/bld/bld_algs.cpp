#include "bld/bld_algs.hpp"
#include "util/wrappers.hpp"
#include <gsl/gsl_sf_psi.h>
#include <iostream>
#include <util/util.hpp>

using namespace bnmf_algs;

/**
 * @brief Do parameter checks on seq_greedy_bld parameters and return error
 * message if there is any.
 *
 * @return Error message. If there isn't any error, returns "".
 */
static std::string seq_greedy_bld_param_checks(
    const matrix_t& X, size_t z,
    const allocation_model::AllocModelParams& model_params) {
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

tensord<3>
bld::seq_greedy_bld(const matrix_t& X, size_t z,
                    const allocation_model::AllocModelParams& model_params) {
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

    tensord<3> S(x, y, z);
    S.setZero();
    matrix_t S_ipk = matrix_t::Zero(x, z); // S_{i+k}
    vector_t S_ppk = vector_t::Zero(z);    // S_{++k}
    matrix_t S_pjk = matrix_t::Zero(y, z); // S_{+jk}

    auto rand_gen = util::make_gsl_rng(gsl_rng_taus);
    int ii, jj;
    vector_t ll(z);
    for (int i = 0; i < (int)sum; ++i) {
        auto rand_discrete = util::make_gsl_ran_discrete(nonzero_indices.size(),
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

std::tuple<matrix_t, matrix_t, vector_t>
bld::bld_fact(const tensord<3>& S,
              const allocation_model::AllocModelParams& model_params,
              double eps) {
    long x = S.dimension(0), y = S.dimension(1), z = S.dimension(2);

    if (model_params.alpha.size() != x) {
        throw std::invalid_argument(
            "Number of alpha parameters must be equal to S.dimension(0)");
    }
    if (model_params.beta.size() != z) {
        throw std::invalid_argument(
            "Number of beta parameters must be equal to S.dimension(1)");
    }

    tensord<2> S_ipk = S.sum(shape<1>({1}));
    tensord<2> S_pjk = S.sum(shape<1>({0}));
    tensord<1> S_pjp = S.sum(shape<2>({0, 2}));

    matrix_t W(x, z);
    matrix_t H(z, y);
    vector_t L(y);

    for (int i = 0; i < x; ++i) {
        for (int k = 0; k < z; ++k) {
            W(i, k) = model_params.alpha[i] + S_ipk(i, k) - 1;
        }
    }
    for (int k = 0; k < z; ++k) {
        for (int j = 0; j < y; ++j) {
            H(k, j) = model_params.beta[k] + S_pjk(j, k) - 1;
        }
    }
    for (int j = 0; j < y; ++j) {
        L(j) = (model_params.a + S_pjp(j) - 1) / (model_params.b + 1 + eps);
    }

    // normalize
    vector_t W_colsum = W.colwise().sum() + vector_t::Constant(W.cols(), eps);
    vector_t H_colsum = H.colwise().sum() + vector_t::Constant(H.cols(), eps);
    W = W.array().rowwise() / W_colsum.array();
    H = H.array().rowwise() / H_colsum.array();

    return std::make_tuple(W, H, L);
};

tensord<3> bld::bld_mult(const matrix_t& X, size_t z,
                         const allocation_model::AllocModelParams& model_params,
                         size_t max_iter, double eps) {
    // todo: think of a way to organize these parameter checks
    if ((X.array() < 0).any()) {
        throw std::invalid_argument("X must be nonnegative");
    }
    if (X.rows() != model_params.alpha.size()) {
        throw std::invalid_argument(
            "Number of rows of X must be equal to number of alpha parameters");
    }
    if (z != model_params.beta.size()) {
        throw std::invalid_argument(
            "z must be equal to number of beta parameters");
    }

    long x = X.rows(), y = X.cols();
    tensord<3> S(x, y, z);

    // initialize tensor S
    {
        auto rand_gen = util::make_gsl_rng(gsl_rng_taus);
        std::vector<double> dirichlet_params(z, 1);
        std::vector<double> dirichlet_variates(z);
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                gsl_ran_dirichlet(rand_gen.get(), z, dirichlet_params.data(),
                                  dirichlet_variates.data());
                for (int k = 0; k < z; ++k) {
                    S(i, j, k) = X(i, j) * dirichlet_variates[k];
                }
            }
        }
    }

    tensord<3> grad_plus(x, y, z);
    matrix_t grad_minus(x, z);
    matrix_t nom_mult(x, y);
    matrix_t denom_mult(x, y);
    tensord<2> S_pjk(y, z); // S_{+jk}
    tensord<2> S_ipk(x, z); // S_{i+k}
    tensord<2> S_ijp(x, y); // S_{ij+}
    matrix_t alpha_eph(x, z);
    matrix_t beta_eph(y, z);
    for (size_t eph = 0; eph < max_iter; ++eph) {
        // update S_pjk, S_ipk, S_ijp
        S_pjk = S.sum(shape<1>({0}));
        S_ipk = S.sum(shape<1>({1}));
        S_ijp = S.sum(shape<1>({2}));
        // update alpha_eph
        for (int i = 0; i < x; ++i) {
            for (int k = 0; k < z; ++k) {
                alpha_eph(i, k) = model_params.alpha[i] + S_ipk(i, k);
            }
        }
        // update beta_eph
        for (int j = 0; j < y; ++j) {
            for (int k = 0; k < z; ++k) {
                beta_eph(j, k) = model_params.beta[k] + S_pjk(j, k);
            }
        }
        // update grad_plus
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                for (int k = 0; k < z; ++k) {
                    grad_plus(i, j, k) =
                        gsl_sf_psi(beta_eph(j, k)) - gsl_sf_psi(S(i, j, k) + 1);
                }
            }
        }
        // update grad_minus
        vector_t alpha_eph_sum = alpha_eph.colwise().sum();
        for (int i = 0; i < x; ++i) {
            for (int k = 0; k < z; ++k) {
                grad_minus(i, k) =
                    gsl_sf_psi(alpha_eph_sum(k)) - gsl_sf_psi(alpha_eph(i, k));
            }
        }
        // update nom_mult, denom_mult
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                double xdiv = 1.0 / (X(i, j) + eps);
                double nom_sum = 0, denom_sum = 0;
                for (int k = 0; k < z; ++k) {
                    nom_sum += (S(i, j, k) * xdiv * grad_minus(i, k));
                    denom_sum += (S(i, j, k) * xdiv * grad_plus(i, j, k));
                }
                nom_mult(i, j) = nom_sum;
                denom_mult(i, j) = denom_sum;
            }
        }
        // update S
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                double s_ij = S_ijp(i, j);
                double x_over_s = X(i, j) / (s_ij + eps);
                for (int k = 0; k < z; ++k) {
                    S(i, j, k) *= (grad_plus(i, j, k) + nom_mult(i, j)) /
                                  (grad_minus(i, k) + denom_mult(i, j) + eps);
                    S(i, j, k) *= x_over_s;
                }
            }
        }
    }

    return S;
}

/**
 * @brief Calculate step size to use during gradient ascent iterations.
 *
 * @param step Current step.
 *
 * @return Step size to use at step n.
 */
static double eta(size_t step) { return 0.1 / std::pow(step + 1, 0.55); }

tensord<3> bld::bld_add(const matrix_t& X, size_t z,
                        const allocation_model::AllocModelParams& model_params,
                        size_t max_iter, double eps) {
    // todo: think of a way to organize these parameter checks
    if ((X.array() < 0).any()) {
        throw std::invalid_argument("X must be nonnegative");
    }
    if (X.rows() != model_params.alpha.size()) {
        throw std::invalid_argument(
            "Number of rows of X must be equal to number of alpha parameters");
    }
    if (z != model_params.beta.size()) {
        throw std::invalid_argument(
            "z must be equal to number of beta parameters");
    }

    long x = X.rows(), y = X.cols();
    tensord<3> S(x, y, z);
    // initialize tensor S
    {
        auto rand_gen = util::make_gsl_rng(gsl_rng_taus);
        std::vector<double> dirichlet_params(z, 1);
        std::vector<double> dirichlet_variates(z);
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                gsl_ran_dirichlet(rand_gen.get(), z, dirichlet_params.data(),
                                  dirichlet_variates.data());
                for (int k = 0; k < z; ++k) {
                    S(i, j, k) = X(i, j) * dirichlet_variates[k];
                }
            }
        }
    }

    tensord<2> S_pjk(y, z); // S_{+jk}
    tensord<2> S_ipk(x, z); // S_{i+k}
    matrix_t alpha_eph(x, z);
    matrix_t beta_eph(y, z);
    tensord<3> grad_S(x, y, z);
    tensord<3> eps_tensor(x, y, z);
    eps_tensor.setConstant(eps);
    for (size_t eph = 0; eph < max_iter; ++eph) {
        // update S_pjk, S_ipk, S_ijp
        S_pjk = S.sum(shape<1>({0}));
        S_ipk = S.sum(shape<1>({1}));
        // update alpha_eph
        for (int i = 0; i < x; ++i) {
            for (int k = 0; k < z; ++k) {
                alpha_eph(i, k) = model_params.alpha[i] + S_ipk(i, k);
            }
        }
        // update beta_eph
        for (int j = 0; j < y; ++j) {
            for (int k = 0; k < z; ++k) {
                beta_eph(j, k) = model_params.beta[k] + S_pjk(j, k);
            }
        }
        // update grad_S
        vector_t alpha_eph_sum = alpha_eph.colwise().sum();
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                for (int k = 0; k < z; ++k) {
                    grad_S(i, j, k) = gsl_sf_psi(beta_eph(j, k)) -
                                      gsl_sf_psi(S(i, j, k) + 1) -
                                      gsl_sf_psi(alpha_eph_sum(k)) +
                                      gsl_sf_psi(alpha_eph(i, k));
                }
            }
        }
        // construct Z
        tensord<3> Z = S.log() + eps_tensor;
        tensord<2> S_mult = (S * grad_S).sum(shape<1>({2}));
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                for (int k = 0; k < z; ++k) {
                    Z(i, j, k) += eta(eph) * S(i, j, k) *
                                  (X(i, j) * grad_S(i, j, k) - S_mult(i, j));
                }
            }
        }
        Z = Z.exp();
        bnmf_algs::util::normalize(Z, 2, bnmf_algs::util::NormType::L1);
        // update S
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                for (int k = 0; k < z; ++k) {
                    S(i, j, k) = X(i, j)*Z(i, j, k);
                }
            }
        }
    }
}
