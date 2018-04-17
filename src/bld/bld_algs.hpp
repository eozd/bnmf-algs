#pragma once

#include "alloc_model/alloc_model_funcs.hpp"
#include "alloc_model/alloc_model_params.hpp"
#include "defs.hpp"
#include "util/generator.hpp"
#include "util/sampling.hpp"
#include "util/util.hpp"
#include <gsl/gsl_sf_psi.h>
#include <omp.h>
#include <thread>

#ifdef USE_CUDA
#include "cuda/memory.hpp"
#include "cuda/tensor_ops.hpp"
#endif

namespace bnmf_algs {
namespace details {

/**
 * @brief Computer type that will compute the next collapsed gibbs sample when
 * its operator() is invoked.
 *
 * CollapsedGibbsComputer will compute a new bnmf_algs::tensord<3> object from
 * the previous sample when its operator() is invoked.
 */
template <typename T, typename Scalar> class CollapsedGibbsComputer {
  public:
    /**
     * @brief Construct a new CollapsedGibbsComputer.
     *
     * @param X Matrix \f$X\f$ of size \f$x \times y\f$ that will be used during
     * sampling.
     * @param z Depth of the output tensor \f$S\f$ with size \f$x \times y
     * \times z\f$.
     * @param model_params Allocation model parameters. See
     * bnmf_algs::alloc_model::Params<double>.
     * @param max_iter Maximum number of iterations.
     * @param eps Floating point epsilon value to be used to prevent division by
     * 0 errors.
     */
    explicit CollapsedGibbsComputer(
        const matrix_t<T>& X, size_t z,
        const alloc_model::Params<Scalar>& model_params, size_t max_iter,
        double eps)
        : model_params(model_params),
          one_sampler_repl(util::sample_ones_replace(X, max_iter)),
          one_sampler_no_repl(util::sample_ones_noreplace(X)),
          U_ipk(matrix_t<T>::Zero(X.rows(), z)), U_ppk(vector_t<T>::Zero(z)),
          U_pjk(matrix_t<T>::Zero(X.cols(), z)),
          sum_alpha(std::accumulate(model_params.alpha.begin(),
                                    model_params.alpha.end(), Scalar())),
          eps(eps), rnd_gen(util::gsl_rng_wrapper(gsl_rng_alloc(gsl_rng_taus),
                                                  gsl_rng_free)) {}

    /**
     * @brief Function call operator that will compute the next tensor sample
     * from the previous sample.
     *
     * After this function exits, the object referenced by S_prev will be
     * modified to store the next sample.
     *
     * Note that CollapsedGibbsComputer object satisfies the
     * bnmf_algs::util::Generator and bnmf_algs::util::ComputationIterator
     * Computer interface. Hence, a sequence of samples may be generated
     * efficiently by using this Computer with bnmf_algs::util::Generator or
     * bnmf_algs::util::ComputationIterator.
     *
     * @param curr_step Current step of collapsed gibbs computation.
     * @param S_prev Previously computed sample to modify in-place.
     */
    void operator()(size_t curr_step, tensor_t<T, 3>& S_prev) {
        size_t i, j;
        if (curr_step == 0) {
            for (const auto& pair : one_sampler_no_repl) {
                std::tie(i, j) = pair;
                increment_sampling(i, j, S_prev);
            }
        } else {
            std::tie(i, j) = *one_sampler_repl.begin();
            ++one_sampler_repl.begin();
            decrement_sampling(i, j, S_prev);
            increment_sampling(i, j, S_prev);
        }
    }

  private:
    /**
     * @brief Sample a single multinomial variable and increment corresponding
     * U_ipk, U_ppk, U_pjk and S_prev entries.
     *
     * This function constructs the probability distribution of each \f$z\f$
     * event in a multinomial distribution, draws a single sample and increments
     * @code S_prev(i, j, k), U_ipk(i, k), U_ppk(k), U_pjk(j, k) @endcode
     * entries.
     *
     * @param i Row of U_ipk and alpha parameter to use.
     * @param j Row of U_pjk to use.
     * @param S_prev Previously computed tensor \f$S\f$.
     */
    void increment_sampling(size_t i, size_t j, tensor_t<T, 3>& S_prev) {
        vector_t<T> prob(U_ppk.cols());
        Eigen::Map<vector_t<T>> beta(model_params.beta.data(), 1,
                                     model_params.beta.size());
        std::vector<unsigned int> multinomial_sample(
            static_cast<unsigned long>(U_ppk.cols()));

        vector_t<T> alpha_row = U_ipk.row(i).array() + model_params.alpha[i];
        vector_t<T> beta_row = (U_pjk.row(j) + beta).array();
        vector_t<T> sum_alpha_row = U_ppk.array() + sum_alpha;
        prob = alpha_row.array() * beta_row.array() /
               (sum_alpha_row.array() + eps);

        gsl_ran_multinomial(rnd_gen.get(), static_cast<size_t>(prob.cols()), 1,
                            prob.data(), multinomial_sample.data());
        auto k = std::distance(multinomial_sample.begin(),
                               std::max_element(multinomial_sample.begin(),
                                                multinomial_sample.end()));

        ++S_prev(i, j, k);
        ++U_ipk(i, k);
        ++U_ppk(k);
        ++U_pjk(j, k);
    }

    /**
     * @brief Sample a single multinomial variable and decremnet corresponding
     * U_ipk, U_ppk, U_pjk and S_prev entries.
     *
     * This function constructs the probability distribution of each \f$z\f$
     * event in a multinomial distribution by using @code S_prev(i, j, :)
     * @endcode fiber, draws a sample from multinomial and decrements
     * @code S_prev(i, j, k), U_ipk(i, k), U_ppk(k), U_pjk(j, k) @endcode
     * entries.
     *
     * @param i Row of S_prev.
     * @param j Column of S_prev.
     * @param S_prev Previously computed tensor \f$S\f$.
     */
    void decrement_sampling(size_t i, size_t j, tensor_t<T, 3>& S_prev) {
        vector_t<T> prob(U_ppk.cols());
        Eigen::Map<vector_t<T>> beta(model_params.beta.data(), 1,
                                     model_params.beta.size());
        std::vector<unsigned int> multinomial_sample(
            static_cast<unsigned long>(U_ppk.cols()));

        // todo: can we take a fiber from S with contiguous memory?
        for (long k = 0; k < S_prev.dimension(2); ++k) {
            prob(k) = S_prev(i, j, k);
        }

        gsl_ran_multinomial(rnd_gen.get(), static_cast<size_t>(prob.cols()), 1,
                            prob.data(), multinomial_sample.data());
        auto k = std::distance(multinomial_sample.begin(),
                               std::max_element(multinomial_sample.begin(),
                                                multinomial_sample.end()));

        --S_prev(i, j, k);
        --U_ipk(i, k);
        --U_ppk(k);
        --U_pjk(j, k);
    }

  private:
    alloc_model::Params<Scalar> model_params;
    // computation variables
  private:
    util::Generator<std::pair<int, int>, details::SampleOnesReplaceComputer<T>>
        one_sampler_repl;
    util::Generator<std::pair<int, int>,
                    details::SampleOnesNoReplaceComputer<T>>
        one_sampler_no_repl;
    matrix_t<T> U_ipk;
    vector_t<T> U_ppk;
    matrix_t<T> U_pjk;
    T sum_alpha;
    double eps;
    util::gsl_rng_wrapper rnd_gen;
};

/**
 * @brief Do parameter checks on BLD computing function parameters and throw an
 * assertion error if the parameters are incorrect.
 *
 * @tparam T Type of the entries of the input matrix X.
 * @tparam Scalar Type of the model parameters.
 *
 * @remark Throws assertion error if the parameters are incorrect.
 */
template <typename T, typename Scalar>
void check_bld_params(const matrix_t<T>& X, size_t z,
                      const alloc_model::Params<Scalar>& model_params) {
    BNMF_ASSERT((X.array() >= 0).all(), "X must be nonnegative");
    BNMF_ASSERT(
        model_params.alpha.size() == static_cast<size_t>(X.rows()),
        "Number of alpha parameters must be equal to number of rows of X");
    BNMF_ASSERT(model_params.beta.size() == static_cast<size_t>(z),
                "Number of beta parameters must be equal to z");
}

template <typename T>
void update_X_hat(const matrix_t<T>& nu, const matrix_t<T>& mu,
                  matrix_t<T>& X_hat) {
    X_hat = nu * mu;
}

template <typename T>
void update_orig_over_appr(const matrix_t<T>& X, const matrix_t<T>& X_hat,
                           const matrix_t<T>& eps_mat,
                           matrix_t<T>& orig_over_appr) {
    orig_over_appr = X.array() / (X_hat + eps_mat).array();
}

template <typename T>
void update_orig_over_appr_squared(const matrix_t<T>& X,
                                   const matrix_t<T>& X_hat,
                                   const matrix_t<T>& eps_mat,
                                   matrix_t<T>& orig_over_appr_squared) {
    matrixd appr_squared = X_hat.array().pow(2);
    orig_over_appr_squared = X.array() / (appr_squared + eps_mat).array();
}

template <typename T>
void update_S(const matrix_t<T>& orig_over_appr, const matrix_t<T>& nu,
              const matrix_t<T>& mu, tensor_t<T, 3>& S) {
    auto x = static_cast<size_t>(S.dimension(0));
    auto y = static_cast<size_t>(S.dimension(1));
    auto z = static_cast<size_t>(S.dimension(2));

    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            for (size_t k = 0; k < z; ++k) {
                S(i, j, k) = orig_over_appr(i, j) * nu(i, k) * mu(k, j);
            }
        }
    }
}

template <typename T>
void update_alpha_eph(const vector_t<T>& alpha, const tensor_t<T, 3>& S,
                      matrix_t<T>& alpha_eph) {
    auto x = static_cast<size_t>(S.dimension(0));
    auto z = static_cast<size_t>(S.dimension(2));

    tensor_t<T, 2> S_ipk_tensor = S.sum(shape<1>({1}));
    Eigen::Map<matrix_t<T>> S_ipk(S_ipk_tensor.data(), x, z);
    alpha_eph = S_ipk.array().colwise() + alpha.transpose().array();
}

template <typename T>
void update_beta_eph(const vector_t<T>& beta, const tensor_t<T, 3>& S,
                     matrix_t<T>& beta_eph) {
    auto y = static_cast<size_t>(S.dimension(1));
    auto z = static_cast<size_t>(S.dimension(2));

    tensor_t<T, 2> S_pjk_tensor = S.sum(shape<1>({0}));
    Eigen::Map<matrix_t<T>> S_pjk(S_pjk_tensor.data(), y, z);
    beta_eph = S_pjk.array().rowwise() + beta.array();
}

template <typename T>
void update_grad_plus(const matrix_t<T>& beta_eph, const tensor_t<T, 3>& S,
                      tensor_t<T, 3>& grad_plus) {
    auto x = static_cast<size_t>(grad_plus.dimension(0));
    auto y = static_cast<size_t>(grad_plus.dimension(1));
    auto z = static_cast<size_t>(grad_plus.dimension(2));

    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            for (size_t k = 0; k < z; ++k) {
                grad_plus(i, j, k) =
                    gsl_sf_psi(beta_eph(j, k)) - gsl_sf_psi(S(i, j, k) + 1);
            }
        }
    }
}

template <typename T>
void update_grad_minus(const matrix_t<T>& alpha_eph, matrix_t<T>& grad_minus) {
    auto x = static_cast<size_t>(grad_minus.rows());
    auto z = static_cast<size_t>(grad_minus.cols());

    vector_t<T> alpha_eph_sum = alpha_eph.colwise().sum();
    for (size_t i = 0; i < x; ++i) {
        for (size_t k = 0; k < z; ++k) {
            grad_minus(i, k) =
                gsl_sf_psi(alpha_eph_sum(k)) - gsl_sf_psi(alpha_eph(i, k));
        }
    }
}

template <typename T>
void update_nu(const matrix_t<T>& orig_over_appr,
               const matrix_t<T>& orig_over_appr_squared, const matrix_t<T>& mu,
               const tensor_t<T, 3>& grad_plus, const matrix_t<T>& grad_minus,
               double eps, matrix_t<T>& nu) {
    auto x = static_cast<size_t>(grad_plus.dimension(0));
    auto y = static_cast<size_t>(grad_plus.dimension(1));
    auto z = static_cast<size_t>(grad_plus.dimension(2));

    matrix_t<T> nom = matrix_t<T>::Zero(x, z);
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            for (size_t k = 0; k < z; ++k) {
                nom(i, k) +=
                    orig_over_appr(i, j) * mu(k, j) * grad_plus(i, j, k);
            }
        }
    }
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            for (size_t k = 0; k < z; ++k) {
                for (size_t c = 0; c < z; ++c) {
                    nom(i, k) += orig_over_appr_squared(i, j) * mu(k, j) *
                                 nu(i, c) * mu(c, j) * grad_minus(i, c);
                }
            }
        }
    }
    matrix_t<T> denom = matrix_t<T>::Zero(x, z);
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            for (size_t k = 0; k < z; ++k) {
                denom(i, k) +=
                    orig_over_appr(i, j) * mu(k, j) * grad_minus(i, k);
            }
        }
    }
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            for (size_t k = 0; k < z; ++k) {
                for (size_t c = 0; c < z; ++c) {
                    denom(i, k) += orig_over_appr_squared(i, j) * mu(k, j) *
                                   nu(i, c) * mu(c, j) * grad_plus(i, j, c);
                }
            }
        }
    }

    nu = nu.array() * nom.array() / (denom.array() + eps);

    // normalize
    vector_t<T> nu_rowsum_plus_eps =
        nu.rowwise().sum() + vector_t<T>::Constant(nu.rows(), eps).transpose();
    nu = nu.array().colwise() / nu_rowsum_plus_eps.transpose().array();
}

template <typename T>
static void update_mu(const matrix_t<T>& orig_over_appr,
                      const matrix_t<T>& orig_over_appr_squared,
                      const matrix_t<T>& nu, const tensor_t<T, 3>& grad_plus,
                      const matrix_t<T>& grad_minus, double eps,
                      matrix_t<T>& mu) {
    auto x = static_cast<size_t>(grad_plus.dimension(0));
    auto y = static_cast<size_t>(grad_plus.dimension(1));
    auto z = static_cast<size_t>(grad_plus.dimension(2));

    matrix_t<T> nom = matrix_t<T>::Zero(z, y);
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            for (size_t k = 0; k < z; ++k) {
                nom(k, j) +=
                    orig_over_appr(i, j) * nu(i, k) * grad_plus(i, j, k);
            }
        }
    }
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            for (size_t k = 0; k < z; ++k) {
                for (size_t c = 0; c < z; ++c) {
                    nom(k, j) += orig_over_appr_squared(i, j) * nu(i, k) *
                                 nu(i, c) * mu(c, j) * grad_minus(i, c);
                }
            }
        }
    }
    matrix_t<T> denom = matrix_t<T>::Zero(z, y);
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            for (size_t k = 0; k < z; ++k) {
                denom(k, j) +=
                    orig_over_appr(i, j) * nu(i, k) * grad_minus(i, k);
            }
        }
    }
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            for (size_t k = 0; k < z; ++k) {
                for (size_t c = 0; c < z; ++c) {
                    denom(k, j) += orig_over_appr_squared(i, j) * nu(i, k) *
                                   nu(i, c) * mu(c, j) * grad_plus(i, j, c);
                }
            }
        }
    }

    mu = mu.array() * nom.array() / (denom.array() + eps);

    // normalize
    vector_t<T> mu_colsum_plus_eps =
        mu.colwise().sum() + vector_t<T>::Constant(mu.cols(), eps);
    mu = mu.array().rowwise() / mu_colsum_plus_eps.array();
}
} // namespace details

/**
 * @brief Namespace that contains solver and auxiliary functions for Best Latent
 * Decomposition (BLD) problem.
 */
namespace bld {
/**
 * @brief Compute tensor \f$S\f$, the solution of BLD problem \cite
 * kurutmazbayesian, from matrix \f$X\f$ using sequential greedy method.
 *
 * According to Allocation Model \cite kurutmazbayesian,
 *
 * \f[
 * L_j \sim \mathcal{G}(a, b) \qquad W_{:k} \sim \mathcal{D}(\alpha) \qquad
 * H_{:j} \sim \mathcal{D}(\beta) \f]
 *
 * Each entry \f$S_{ijk} \sim \mathcal{PO}(W_{ik}H_{kj}L_j)\f$ and overall \f$X
 * = S_{ij+}\f$.
 *
 * In this context, Best Latent Decomposition (BLD) problem is \cite
 * kurutmazbayesian,
 *
 * \f[
 * S^* = \underset{S_{::+}=X}{\arg \max}\text{ }p(S).
 * \f]
 *
 * Sequential greedy BLD algorithm works by greedily allocating each entry of
 * input matrix, \f$X_{ij}\f$, one by one to its corresponding bin
 * \f$S_{ijk}\f$. \f$k\f$ is chosen by considering the increase in log
 * marginal value of tensor \f$S\f$ when \f$S_{ijk}\f$ is increased by \f$1\f$.
 * At a given time, \f$i, j\f$ is chosen as the indices of the largest entry of
 * matrix \f$X\f$.
 *
 * @param X Nonnegative matrix of size \f$x \times y\f$ to decompose.
 * @param z Number of matrices into which matrix \f$X\f$ will be decomposed.
 * This is the depth of the output tensor \f$S\f$.
 * @param model_params Allocation model parameters. See
 * bnmf_algs::alloc_model::Params<double>.
 *
 * @return Tensor \f$S\f$ of size \f$x \times y \times z\f$ where \f$X =
 * S_{ij+}\f$.
 *
 * @throws std::invalid_argument if matrix \f$X\f$ is not nonnegative, if size
 * of alpha is not \f$x\f$, if size of beta is not \f$z\f$.
 *
 * @remark Worst-case time complexity is \f$O(xy +
 * z(\log{xy})\sum_{ij}X_{ij})\f$ for a matrix \f$X_{x \times y}\f$ and output
 * tensor \f$S_{x \times y \times z}\f$.
 */
template <typename T, typename Scalar>
tensor_t<T, 3> seq_greedy_bld(const matrix_t<T>& X, size_t z,
                              const alloc_model::Params<Scalar>& model_params) {
    details::check_bld_params(X, z, model_params);

    long x = X.rows();
    long y = X.cols();

    // exit early if sum is 0
    if (std::abs(X.sum()) <= std::numeric_limits<double>::epsilon()) {
        tensor_t<T, 3> S(x, y, z);
        S.setZero();
        return S;
    }

    // variables to update during the iterations
    tensor_t<T, 3> S(x, y, z);
    S.setZero();
    matrix_t<T> S_ipk = matrix_t<T>::Zero(x, z); // S_{i+k}
    vector_t<T> S_ppk = vector_t<T>::Zero(z);    // S_{++k}
    matrix_t<T> S_pjk = matrix_t<T>::Zero(y, z); // S_{+jk}
    Scalar sum_alpha = std::accumulate(model_params.alpha.begin(),
                                       model_params.alpha.end(), 0.0);

    // sample X matrix one by one
    auto matrix_sampler = util::sample_ones_noreplace(X);
    int ii, jj;
    for (const auto& sample : matrix_sampler) {
        std::tie(ii, jj) = sample;

        // find the bin that increases log marginal the largest when it is
        // incremented
        size_t kmax = 0;
        T curr_max = std::numeric_limits<double>::lowest();
        for (size_t k = 0; k < z; ++k) {
            T log_marginal_change =
                std::log(model_params.alpha[ii] + S_ipk(ii, k)) -
                std::log(sum_alpha + S_ppk(k)) - std::log(1 + S(ii, jj, k)) +
                std::log(model_params.beta[k] + S_pjk(jj, k));

            if (log_marginal_change > curr_max) {
                curr_max = log_marginal_change;
                kmax = k;
            }
        }

        // increase the corresponding bin accumulators
        ++S(ii, jj, kmax);
        ++S_ipk(ii, kmax);
        ++S_ppk(kmax);
        ++S_pjk(jj, kmax);
    }

    return S;
}

/**
 * @brief Compute matrices \f$W, H\f$ and vector \f$L\f$ from tensor \f$S\f$
 * according to the allocation model.
 *
 * According to Allocation Model \cite kurutmazbayesian,
 *
 * \f[
 * L_j \sim \mathcal{G}(a, b) \qquad W_{:k} \sim \mathcal{D}(\alpha) \qquad
 * H_{:j} \sim \mathcal{D}(\beta) \f]
 *
 * Each entry \f$S_{ijk} \sim \mathcal{PO}(W_{ik}H_{kj}L_j)\f$.
 *
 * @param S Tensor \f$S\f$ of shape \f$x \times y \times z\f$.
 * @param model_params Allocation model parameters. See
 * bnmf_algs::alloc_model::Params<double>.
 * @param eps Floating point epsilon value to be used to prevent division by 0
 * errors.
 *
 * @return std::tuple of matrices \f$W_{x \times z}, H_{z \times y}\f$ and
 * vector \f$L_y\f$.
 */
template <typename T, typename Scalar>
std::tuple<matrix_t<T>, matrix_t<T>, vector_t<T>>
bld_fact(const tensor_t<T, 3>& S,
         const alloc_model::Params<Scalar>& model_params, double eps = 1e-50) {
    auto x = static_cast<size_t>(S.dimension(0));
    auto y = static_cast<size_t>(S.dimension(1));
    auto z = static_cast<size_t>(S.dimension(2));

    BNMF_ASSERT(model_params.alpha.size() == x,
                "Number of alpha parameters must be equal to S.dimension(0)");
    BNMF_ASSERT(model_params.beta.size() == z,
                "Number of beta parameters must be equal to z");

    tensor_t<T, 2> S_ipk = S.sum(shape<1>({1}));
    tensor_t<T, 2> S_pjk = S.sum(shape<1>({0}));
    tensor_t<T, 1> S_pjp = S.sum(shape<2>({0, 2}));

    matrix_t<T> W(x, z);
    matrix_t<T> H(z, y);
    vector_t<T> L(y);

    for (size_t i = 0; i < x; ++i) {
        for (size_t k = 0; k < z; ++k) {
            W(i, k) = model_params.alpha[i] + S_ipk(i, k) - 1;
        }
    }
    for (size_t k = 0; k < z; ++k) {
        for (size_t j = 0; j < y; ++j) {
            H(k, j) = model_params.beta[k] + S_pjk(j, k) - 1;
        }
    }
    for (size_t j = 0; j < y; ++j) {
        L(j) = (model_params.a + S_pjp(j) - 1) / (model_params.b + 1 + eps);
    }

    // normalize
    vector_t<T> W_colsum =
        W.colwise().sum() + vector_t<T>::Constant(W.cols(), eps);
    vector_t<T> H_colsum =
        H.colwise().sum() + vector_t<T>::Constant(H.cols(), eps);
    W = W.array().rowwise() / W_colsum.array();
    H = H.array().rowwise() / H_colsum.array();

    return std::make_tuple(W, H, L);
}

/**
 * @brief Compute tensor \f$S\f$, the solution of BLD problem \cite
 * kurutmazbayesian, from matrix \f$X\f$ using multiplicative update rules.
 *
 * According to Allocation Model \cite kurutmazbayesian,
 *
 * \f[
 * L_j \sim \mathcal{G}(a, b) \qquad W_{:k} \sim \mathcal{D}(\alpha) \qquad
 * H_{:j} \sim \mathcal{D}(\beta) \f]
 *
 * Each entry \f$S_{ijk} \sim \mathcal{PO}(W_{ik}H_{kj}L_j)\f$ and overall \f$X
 * = S_{ij+}\f$.
 *
 * In this context, Best Latent Decomposition (BLD) problem is \cite
 * kurutmazbayesian,
 *
 * \f[
 * S^* = \underset{S_{::+}=X}{\arg \max}\text{ }p(S).
 * \f]
 *
 * Multiplicative BLD algorithm solves a constrained optimization problem over
 * the set of tensors whose sum over their third index is equal to \f$X\f$.
 * Let's denote this set with \f$\Omega_X = \{S | S_{ij+} = X_{ij}\}\f$. To make
 * the optimization easier the constraints on the found tensors are relaxed by
 * searching over \f$\mathcal{R}^{x \times y \times z}\f$ and defining a
 * projection function \f$\phi : \mathcal{R}^{x \times y \times z} \rightarrow
 * \Omega_X\f$ such that \f$\phi(Z) = S\f$ where \f$Z\f$ is the solution of the
 * unconstrained problem and \f$S\f$ is the solution of the original constrained
 * problem.
 *
 * In this algorithm, \f$\phi\f$ is chosen such that
 *
 * \f[
 *     S_{ijk} = \phi(Z_{ijk}) = \frac{Z_{ijk}X_{ij}}{\sum_{k'}Z_{ijk'}}
 * \f]
 *
 * This particular choice of \f$\phi\f$ leads to multiplicative update rules
 * implemented in this algorithm.
 *
 * @param X Nonnegative matrix of size \f$x \times y\f$ to decompose.
 * @param z Number of matrices into which matrix \f$X\f$ will be decomposed.
 * This is the depth of the output tensor \f$S\f$.
 * @param model_params Allocation model parameters. See
 * bnmf_algs::alloc_model::Params<double>.
 * @param max_iter Maximum number of iterations.
 * @param use_psi_appr If true, use util::psi_appr function to compute the
 * digamma function approximately. If false, compute the digamma function
 * exactly.
 * @param eps Floating point epsilon value to be used to prevent division by 0
 * errors.
 *
 * @return Tensor \f$S\f$ of size \f$x \times y \times z\f$ where \f$X =
 * S_{ij+}\f$.
 *
 * @throws std::invalid_argument if X contains negative entries,
 * if number of rows of X is not equal to number of alpha parameters, if z is
 * not equal to number of beta parameters.
 */
template <typename T, typename Scalar>
tensor_t<T, 3> bld_mult(const matrix_t<T>& X, size_t z,
                        const alloc_model::Params<Scalar>& model_params,
                        size_t max_iter = 1000, bool use_psi_appr = false,
                        double eps = 1e-50) {
    details::check_bld_params(X, z, model_params);

#ifdef USE_OPENMP
    const int num_threads = std::thread::hardware_concurrency();
#endif

    auto x = static_cast<size_t>(X.rows());
    auto y = static_cast<size_t>(X.cols());

    // initialize tensor S
    tensor_t<T, 3> S(x, y, z);
    vectord dirichlet_params = vectord::Constant(z, 1);
    #pragma omp parallel
    {

#ifdef USE_OPENMP
        int thread_id = omp_get_thread_num();
        bool last_thread = thread_id + 1 == num_threads;
        long num_rows = x / num_threads;
        long beg = thread_id * num_rows;
        long end = !last_thread ? beg + num_rows : x;
#else
        long beg = 0;
        long end = x;
#endif

        util::gsl_rng_wrapper rand_gen(gsl_rng_alloc(gsl_rng_taus),
                                       gsl_rng_free);
        vectord dirichlet_variates(z);
        for (long i = beg; i < end; ++i) {
            for (size_t j = 0; j < y; ++j) {
                gsl_ran_dirichlet(rand_gen.get(), z, dirichlet_params.data(),
                                  dirichlet_variates.data());
                for (size_t k = 0; k < z; ++k) {
                    S(i, j, k) = X(i, j) * dirichlet_variates(k);
                }
            }
        }
    }

#ifdef USE_CUDA
    cuda::HostMemory2D<const T> X_host(X.data(), x, y);
    cuda::DeviceMemory2D<T> X_device(x, y);
    cuda::copy2D(X_device, X_host, cudaMemcpyHostToDevice);
#endif

    tensor_t<T, 3> grad_plus(x, y, z);
    matrix_t<T> grad_minus(x, z);
    matrix_t<T> nom_mult(x, y);
    matrix_t<T> denom_mult(x, y);
    matrix_t<T> alpha_eph(x, z);
    vector_t<T> alpha_eph_sum(z);
    matrix_t<T> beta_eph(y, z);
    tensor_t<T, 2> S_pjk_tensor(y, z);
    tensor_t<T, 2> S_ipk_tensor(x, z);
    tensor_t<T, 2> S_ijp_tensor(x, y);

    Eigen::Map<const vector_t<Scalar>> alpha(model_params.alpha.data(),
                                             model_params.alpha.size());
    Eigen::Map<const vector_t<Scalar>> beta(model_params.beta.data(),
                                            model_params.beta.size());

#ifdef USE_OPENMP
    Eigen::ThreadPool tp(num_threads);
    Eigen::ThreadPoolDevice thread_dev(&tp, num_threads);
#endif

    const auto psi_fn = use_psi_appr ? util::psi_appr<T> : gsl_sf_psi;
    for (size_t eph = 0; eph < max_iter; ++eph) {
        // update S_pjk, S_ipk, S_ijp
#ifdef USE_CUDA
        const auto tensor_sums = cuda::tensor_sums(S);
        S_pjk_tensor = tensor_sums[0];
        S_ipk_tensor = tensor_sums[1];
        S_ijp_tensor = tensor_sums[2];
#elif USE_OPENMP
        S_pjk_tensor.device(thread_dev) = S.sum(shape<1>({0}));
        S_ipk_tensor.device(thread_dev) = S.sum(shape<1>({1}));
        S_ijp_tensor.device(thread_dev) = S.sum(shape<1>({2}));
#else
        S_pjk_tensor = S.sum(shape<1>({0}));
        S_ipk_tensor = S.sum(shape<1>({1}));
        S_ijp_tensor = S.sum(shape<1>({2}));
#endif

        Eigen::Map<matrix_t<T>> S_pjk(S_pjk_tensor.data(), y, z);
        Eigen::Map<matrix_t<T>> S_ipk(S_ipk_tensor.data(), x, z);
        Eigen::Map<matrix_t<T>> S_ijp(S_ijp_tensor.data(), x, y);

        // update alpha_eph, beta_eph
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < z; ++k) {
            for (size_t i = 0; i < x; ++i) {
                alpha_eph(i, k) = S_ipk(i, k) + alpha(i);
            }

            for (size_t j = 0; j < y; ++j) {
                beta_eph(j, k) = S_pjk(j, k) + beta(k);
            }
        }

#ifdef USE_CUDA
        cuda::bld_mult::update_grad_plus(S, beta_eph, grad_plus);
#else
        // update grad_plus
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                for (size_t k = 0; k < z; ++k) {
                    grad_plus(i, j, k) =
                        psi_fn(beta_eph(j, k)) - psi_fn(S(i, j, k) + 1);
                }
            }
        }
#endif

        // update alpha_eph_sum
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < z; ++k) {
            alpha_eph_sum(k) = 0;
            for (size_t i = 0; i < x; ++i) {
                alpha_eph_sum(k) += alpha_eph(i, k);
            }
        }

        // update grad_minus
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < x; ++i) {
            for (size_t k = 0; k < z; ++k) {
                grad_minus(i, k) =
                    psi_fn(alpha_eph_sum(k)) - psi_fn(alpha_eph(i, k));
            }
        }

        // update nom_mult, denom_mult
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < x; ++i) {
            for (size_t j = 0; j < y; ++j) {
                T xdiv = 1.0 / (X(i, j) + eps);
                T nom_sum = 0, denom_sum = 0;
                for (size_t k = 0; k < z; ++k) {
                    nom_sum += (S(i, j, k) * xdiv * grad_minus(i, k));
                    denom_sum += (S(i, j, k) * xdiv * grad_plus(i, j, k));
                }
                nom_mult(i, j) = nom_sum;
                denom_mult(i, j) = denom_sum;
            }
        }

        // update S
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < x; ++i) {
            for (size_t j = 0; j < y; ++j) {
                T s_ij = S_ijp(i, j);
                T x_over_s = X(i, j) / (s_ij + eps);
                for (size_t k = 0; k < z; ++k) {
                    S(i, j, k) *= (grad_plus(i, j, k) + nom_mult(i, j)) *
                                  x_over_s /
                                  (grad_minus(i, k) + denom_mult(i, j) + eps);
                }
            }
        }
    }

    return S;
}

/**
 * @brief Compute tensor \f$S\f$, the solution of BLD problem \cite
 * kurutmazbayesian, from matrix \f$X\f$ using additive gradient ascent updates.
 *
 * According to Allocation Model \cite kurutmazbayesian,
 *
 * \f[
 * L_j \sim \mathcal{G}(a, b) \qquad W_{:k} \sim \mathcal{D}(\alpha) \qquad
 * H_{:j} \sim \mathcal{D}(\beta) \f]
 *
 * Each entry \f$S_{ijk} \sim \mathcal{PO}(W_{ik}H_{kj}L_j)\f$ and overall \f$X
 * = S_{ij+}\f$.
 *
 * In this context, Best Latent Decomposition (BLD) problem is \cite
 * kurutmazbayesian,
 *
 * \f[
 * S^* = \underset{S_{::+}=X}{\arg \max}\text{ }p(S).
 * \f]
 *
 * Additive BLD algorithm solves a constrained optimization problem over
 * the set of tensors whose sum over their third index is equal to \f$X\f$.
 * Let's denote this set with \f$\Omega_X = \{S | S_{ij+} = X_{ij}\}\f$. To make
 * the optimization easier the constraints on the found tensors are relaxed by
 * searching over \f$\mathcal{R}^{x \times y \times z}\f$ and defining a
 * projection function \f$\phi : \mathcal{R}^{x \times y \times z} \rightarrow
 * \Omega_X\f$ such that \f$\phi(Z) = S\f$ where \f$Z\f$ is the solution of the
 * unconstrained problem and \f$S\f$ is the solution of the original constrained
 * problem.
 *
 * In this algorithm, \f$\phi\f$ is chosen such that
 *
 * \f[
 *     S_{ijk} = \phi(Z_{ijk}) =
 * X_{ij}\frac{\exp(Z_{ijk})}{\sum_{k'}\exp(Z_{ijk'})} \f]
 *
 * This particular choice of \f$\phi\f$ leads to additive update rules
 * implemented in this algorithm.
 *
 * @param X Nonnegative matrix of size \f$x \times y\f$ to decompose.
 * @param z Number of matrices into which matrix \f$X\f$ will be decomposed.
 * This is the depth of the output tensor \f$S\f$.
 * @param model_params Allocation model parameters. See
 * bnmf_algs::alloc_model::Params<double>.
 * @param max_iter Maximum number of iterations.
 * @param eps Floating point epsilon value to be used to prevent division by 0
 * errors.
 *
 * @return Tensor \f$S\f$ of size \f$x \times y \times z\f$ where \f$X =
 * S_{ij+}\f$.
 *
 * @throws std::invalid_argument if X contains negative entries,
 * if number of rows of X is not equal to number of alpha parameters, if z is
 * not equal to number of beta parameters.
 */
template <typename T, typename Scalar>
tensor_t<T, 3> bld_add(const matrix_t<T>& X, size_t z,
                       const alloc_model::Params<Scalar>& model_params,
                       size_t max_iter = 1000, double eps = 1e-50) {
    details::check_bld_params(X, z, model_params);

    long x = X.rows(), y = X.cols();
    tensor_t<T, 3> S(x, y, z);
    // initialize tensor S
    {
        util::gsl_rng_wrapper rand_gen(gsl_rng_alloc(gsl_rng_taus),
                                       gsl_rng_free);
        std::vector<double> dirichlet_params(z, 1);
        std::vector<double> dirichlet_variates(z);
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                gsl_ran_dirichlet(rand_gen.get(), z, dirichlet_params.data(),
                                  dirichlet_variates.data());
                for (size_t k = 0; k < z; ++k) {
                    S(i, j, k) = X(i, j) * dirichlet_variates[k];
                }
            }
        }
    }

    tensor_t<T, 2> S_pjk(y, z); // S_{+jk}
    tensor_t<T, 2> S_ipk(x, z); // S_{i+k}
    matrix_t<T> alpha_eph(x, z);
    matrix_t<T> beta_eph(y, z);
    tensor_t<T, 3> grad_S(x, y, z);
    tensor_t<T, 3> eps_tensor(x, y, z);
    eps_tensor.setConstant(eps);

    const auto eta = [](const size_t step) -> double {
        return 0.1 / std::pow(step + 1, 0.55);
    };

    for (size_t eph = 0; eph < max_iter; ++eph) {
        // update S_pjk, S_ipk, S_ijp
        S_pjk = S.sum(shape<1>({0}));
        S_ipk = S.sum(shape<1>({1}));
        // update alpha_eph
        for (int i = 0; i < x; ++i) {
            for (size_t k = 0; k < z; ++k) {
                alpha_eph(i, k) = model_params.alpha[i] + S_ipk(i, k);
            }
        }
        // update beta_eph
        for (int j = 0; j < y; ++j) {
            for (size_t k = 0; k < z; ++k) {
                beta_eph(j, k) = model_params.beta[k] + S_pjk(j, k);
            }
        }
        // update grad_S
        vector_t<T> alpha_eph_sum = alpha_eph.colwise().sum();
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                for (size_t k = 0; k < z; ++k) {
                    grad_S(i, j, k) = gsl_sf_psi(beta_eph(j, k)) -
                                      gsl_sf_psi(S(i, j, k) + 1) -
                                      gsl_sf_psi(alpha_eph_sum(k)) +
                                      gsl_sf_psi(alpha_eph(i, k));
                }
            }
        }
        // construct Z
        tensor_t<T, 3> Z = S.log() + eps_tensor;
        tensor_t<T, 2> S_mult = (S * grad_S).sum(shape<1>({2}));
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                for (size_t k = 0; k < z; ++k) {
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
                for (size_t k = 0; k < z; ++k) {
                    S(i, j, k) = X(i, j) * Z(i, j, k);
                }
            }
        }
    }
    return S;
}

/**
 * @brief Compute a sequence of \f$S\f$ tensors using Collapsed Gibbs Sampler
 * method.
 *
 * According to Allocation Model \cite kurutmazbayesian,
 *
 * \f[
 * L_j \sim \mathcal{G}(a, b) \qquad W_{:k} \sim \mathcal{D}(\alpha) \qquad
 * H_{:j} \sim \mathcal{D}(\beta) \f]
 *
 * Each entry \f$S_{ijk} \sim \mathcal{PO}(W_{ik}H_{kj}L_j)\f$ and overall \f$X
 * = S_{ij+}\f$.
 *
 * In this context, Best Latent Decomposition (BLD) problem is \cite
 * kurutmazbayesian,
 *
 * \f[
 * S^* = \underset{S_{::+}=X}{\arg \max}\text{ }p(S).
 * \f]
 *
 * \todo Explain collapsed gibbs sampler algorithm.
 *
 * @param X Nonnegative matrix of size \f$x \times y\f$ to decompose.
 * @param z Number of matrices into which matrix \f$X\f$ will be decomposed.
 * This is the depth of the output tensor \f$S\f$.
 * @param model_params Allocation model parameters. See
 * bnmf_algs::alloc_model::Params<double>.
 * @param max_iter Maximum number of iterations.
 * @param eps Floating point epsilon value to be used to prevent division by 0
 * errors.
 *
 * @return util::Generator object that will generate a sequence of \f$S\f$
 * tensors using details::CollapsedGibbsComputer as its Computer type.
 *
 * @throws std::invalid_argument if X contains negative entries,
 * if number of rows of X is not equal to number of alpha parameters, if z is
 * not equal to number of beta parameters.
 */
template <typename T, typename Scalar>
util::Generator<tensor_t<T, 3>, details::CollapsedGibbsComputer<T, Scalar>>
collapsed_gibbs(const matrix_t<T>& X, size_t z,
                const alloc_model::Params<Scalar>& model_params,
                size_t max_iter = 1000, double eps = 1e-50) {
    details::check_bld_params(X, z, model_params);

    tensor_t<T, 3> init_val(X.rows(), X.cols(), z);
    init_val.setZero();

    util::Generator<tensor_t<T, 3>, details::CollapsedGibbsComputer<T, Scalar>>
        gen(init_val, max_iter + 2,
            details::CollapsedGibbsComputer<T, Scalar>(X, z, model_params,
                                                       max_iter, eps));

    ++gen.begin();

    return gen;
}

/**
 * @brief Compute tensor \f$S\f$, the solution of BLD problem \cite
 * kurutmazbayesian, from matrix \f$X\f$ using collapsed iterated conditional
 * modes.
 *
 * According to Allocation Model \cite kurutmazbayesian,
 *
 * \f[
 * L_j \sim \mathcal{G}(a, b) \qquad W_{:k} \sim \mathcal{D}(\alpha) \qquad
 * H_{:j} \sim \mathcal{D}(\beta) \f]
 *
 * Each entry \f$S_{ijk} \sim \mathcal{PO}(W_{ik}H_{kj}L_j)\f$ and overall \f$X
 * = S_{ij+}\f$.
 *
 * In this context, Best Latent Decomposition (BLD) problem is \cite
 * kurutmazbayesian,
 *
 * \f[
 * S^* = \underset{S_{::+}=X}{\arg \max}\text{ }p(S).
 * \f]
 *
 * \todo Explain collapsed icm algorithm.
 *
 * @param X Nonnegative matrix of size \f$x \times y\f$ to decompose.
 * @param z Number of matrices into which matrix \f$X\f$ will be decomposed.
 * This is the depth of the output tensor \f$S\f$.
 * @param model_params Allocation model parameters. See
 * bnmf_algs::alloc_model::Params<double>.
 * @param max_iter Maximum number of iterations.
 * @param eps Floating point epsilon value to be used to prevent division by 0
 * errors.
 *
 * @return Tensor \f$S\f$ of size \f$x \times y \times z\f$ where \f$X =
 * S_{ij+}\f$.
 *
 * @throws std::invalid_argument if X contains negative entries,
 * if number of rows of X is not equal to number of alpha parameters, if z is
 * not equal to number of beta parameters.
 */
template <typename T, typename Scalar>
tensor_t<T, 3> collapsed_icm(const matrix_t<T>& X, size_t z,
                             const alloc_model::Params<Scalar>& model_params,
                             size_t max_iter = 1000, double eps = 1e-50) {
    details::check_bld_params(X, z, model_params);

    const auto x = static_cast<size_t>(X.rows());
    const auto y = static_cast<size_t>(X.cols());

    tensor_t<T, 3> U(x, y, z);
    U.setZero();

    matrix_t<T> U_ipk = matrix_t<T>::Zero(x, z);
    vector_t<T> U_ppk = vector_t<T>::Zero(z);
    matrix_t<T> U_pjk = matrix_t<T>::Zero(y, z);

    const Scalar sum_alpha = std::accumulate(model_params.alpha.begin(),
                                             model_params.alpha.end(), 0.0);

    vector_t<T> prob(U_ppk.cols());
    Eigen::Map<const vector_t<Scalar>> beta(model_params.beta.data(), 1,
                                            model_params.beta.size());

    // initializing increment sampling without multinomial
    size_t i, j;
    for (const auto& pair : util::sample_ones_noreplace(X)) {
        std::tie(i, j) = pair;

        vector_t<T> alpha_row = U_ipk.row(i).array() + model_params.alpha[i];
        vector_t<T> beta_row = (U_pjk.row(j) + beta).array();
        vector_t<T> sum_alpha_row = U_ppk.array() + sum_alpha;
        prob = alpha_row.array() * beta_row.array() /
               (sum_alpha_row.array() + eps);

        auto k = std::distance(
            prob.data(),
            std::max_element(prob.data(), prob.data() + prob.cols()));

        ++U(i, j, k);
        ++U_ipk(i, k);
        ++U_ppk(k);
        ++U_pjk(j, k);
    }

    std::vector<unsigned int> multinomial_sample(
        static_cast<unsigned long>(U_ppk.cols()));
    util::gsl_rng_wrapper rnd_gen(gsl_rng_alloc(gsl_rng_taus), gsl_rng_free);

    for (const auto& pair : util::sample_ones_replace(X, max_iter)) {
        std::tie(i, j) = pair;

        // decrement sampling
        for (long k = 0; k < U.dimension(2); ++k) {
            prob(k) = U(i, j, k);
        }

        gsl_ran_multinomial(rnd_gen.get(), static_cast<size_t>(prob.cols()), 1,
                            prob.data(), multinomial_sample.data());
        auto k = std::distance(multinomial_sample.begin(),
                               std::max_element(multinomial_sample.begin(),
                                                multinomial_sample.end()));

        --U(i, j, k);
        --U_ipk(i, k);
        --U_ppk(k);
        --U_pjk(j, k);

        // increment sampling
        vector_t<T> alpha_row = U_ipk.row(i).array() + model_params.alpha[i];
        vector_t<T> beta_row = (U_pjk.row(j) + beta).array();
        vector_t<T> sum_alpha_row = U_ppk.array() + sum_alpha;
        prob = alpha_row.array() * beta_row.array() /
               (sum_alpha_row.array() + eps);

        k = std::distance(
            prob.data(),
            std::max_element(prob.data(), prob.data() + prob.cols()));

        ++U(i, j, k);
        ++U_ipk(i, k);
        ++U_ppk(k);
        ++U_pjk(j, k);
    }

    return U;
}

/* ===================== bld_appr IMPLEMENTATION ============================ */

/**
 * @brief Compute tensor \f$S\f$, the solution of BLD problem \cite
 * kurutmazbayesian, and optimization matrices \f$\mu\f$ and \f$\nu\f$ from
 * matrix \f$X\f$ using the approximate multiplicative algorithm given in
 * \cite kurutmazbayesian.
 *
 * According to Allocation Model \cite kurutmazbayesian,
 *
 * \f[
 * L_j \sim \mathcal{G}(a, b) \qquad W_{:k} \sim \mathcal{D}(\alpha) \qquad
 * H_{:j} \sim \mathcal{D}(\beta) \f]
 *
 * Each entry \f$S_{ijk} \sim \mathcal{PO}(W_{ik}H_{kj}L_j)\f$ and overall \f$X
 * = S_{ij+}\f$.
 *
 * In this context, Best Latent Decomposition (BLD) problem is \cite
 * kurutmazbayesian,
 *
 * \f[
 * S^* = \underset{S_{::+}=X}{\arg \max}\text{ }p(S).
 * \f]
 *
 * This algorithm finds the optimal \f$S\f$ using the multiplicative algorithm
 * employing Lagrange multipliers as given in \cite kurutmazbayesian:
 *
 * \f[
 * S_{ijk} = X_{ij}\frac{\nu_{ik}\mu_{kj}}{\sum_c \nu_{ic}\mu{cj}}.
 * \f]
 *
 * @param X Nonnegative matrix of size \f$x \times y\f$ to decompose.
 * @param z Number of matrices into which matrix \f$X\f$ will be decomposed.
 * This is the depth of the output tensor \f$S\f$.
 * @param model_params Allocation model parameters. See
 * bnmf_algs::alloc_model::Params<double>.
 * @param max_iter Maximum number of iterations.
 * @param eps Floating point epsilon value to be used to prevent division by 0
 * errors.
 *
 * @return std::tuple of tensor \f$S\f$ of size \f$x \times y \times z\f$ where
 * \f$X = S_{ij+}\f$, and matrices \f$\nu\f$ and \f$\mu\f$.
 *
 * @throws std::invalid_argument if X contains negative entries,
 * if number of rows of X is not equal to number of alpha parameters, if z is
 * not equal to number of beta parameters.
 */
template <typename T, typename Scalar>
std::tuple<tensor_t<T, 3>, matrix_t<T>, matrix_t<T>>
bld_appr(const matrix_t<T>& X, size_t z,
         const alloc_model::Params<Scalar>& model_params,
         size_t max_iter = 1000, double eps = 1e-50) {
    details::check_bld_params(X, z, model_params);

    auto x = static_cast<size_t>(X.rows());
    auto y = static_cast<size_t>(X.cols());

    vector_t<Scalar> alpha(model_params.alpha.size());
    vector_t<Scalar> beta(model_params.beta.size());
    std::copy(model_params.alpha.begin(), model_params.alpha.end(),
              alpha.data());
    std::copy(model_params.beta.begin(), model_params.beta.end(), beta.data());

    matrix_t<T> nu(x, z);
    matrix_t<T> mu(y, z);
    // initialize nu and mu using Dirichlet
    {
        util::gsl_rng_wrapper rnd_gen(gsl_rng_alloc(gsl_rng_taus),
                                      gsl_rng_free);
        std::vector<double> dirichlet_params(z, 1);

        // remark: following code depends on the fact that matrixd is row
        // major
        for (size_t i = 0; i < x; ++i) {
            gsl_ran_dirichlet(rnd_gen.get(), z, dirichlet_params.data(),
                              nu.data() + i * z);
        }
        for (size_t j = 0; j < y; ++j) {
            gsl_ran_dirichlet(rnd_gen.get(), z, dirichlet_params.data(),
                              mu.data() + j * z);
        }

        mu.transposeInPlace();
    }

    tensor_t<T, 3> grad_plus(x, y, z);
    matrix_t<T> grad_minus(x, z);
    tensor_t<T, 3> S(x, y, z);
    matrix_t<T> eps_mat = matrix_t<T>::Constant(x, y, eps);
    matrix_t<T> X_hat, orig_over_appr, orig_over_appr_squared, alpha_eph,
        beta_eph;

    for (size_t eph = 0; eph < max_iter; ++eph) {
        // update helpers
        details::update_X_hat(nu, mu, X_hat);
        details::update_orig_over_appr(X, X_hat, eps_mat, orig_over_appr);
        details::update_orig_over_appr_squared(X, X_hat, eps_mat,
                                               orig_over_appr_squared);
        details::update_S(orig_over_appr, nu, mu, S);
        details::update_alpha_eph(alpha, S, alpha_eph);
        details::update_beta_eph(beta, S, beta_eph);
        details::update_grad_plus(beta_eph, S, grad_plus);
        details::update_grad_minus(alpha_eph, grad_minus);

        // update nu
        details::update_nu(orig_over_appr, orig_over_appr_squared, mu,
                           grad_plus, grad_minus, eps, nu);

        // update helpers
        details::update_X_hat(nu, mu, X_hat);
        details::update_orig_over_appr(X, X_hat, eps_mat, orig_over_appr);
        details::update_orig_over_appr_squared(X, X_hat, eps_mat,
                                               orig_over_appr_squared);
        details::update_S(orig_over_appr, nu, mu, S);
        details::update_alpha_eph(alpha, S, alpha_eph);
        details::update_beta_eph(beta, S, beta_eph);
        details::update_grad_plus(beta_eph, S, grad_plus);
        details::update_grad_minus(alpha_eph, grad_minus);

        // update mu
        details::update_mu(orig_over_appr, orig_over_appr_squared, nu,
                           grad_plus, grad_minus, eps, mu);
    }

    details::update_X_hat(nu, mu, X_hat);
    details::update_orig_over_appr(X, X_hat, eps_mat, orig_over_appr);
    details::update_S(orig_over_appr, nu, mu, S);

    return std::make_tuple(S, nu, mu);
}
} // namespace bld
} // namespace bnmf_algs
