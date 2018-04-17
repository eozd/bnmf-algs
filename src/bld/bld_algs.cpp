#include "bld/bld_algs.hpp"
#include "util/sampling.hpp"
#include "util/util.hpp"
#include <gsl/gsl_sf_psi.h>
#include <omp.h>
#include <thread>

#ifdef USE_CUDA
#include "cuda/memory.hpp"
#include "cuda/tensor_ops.hpp"
#endif

using namespace bnmf_algs;

/**
 * @brief Do parameter checks on BLD computing function parameters and return
 * error message if there is any.
 *
 * @return Error message. If there isn't any error, returns "".
 */
static std::string
check_bld_params(const matrixd& X, size_t z,
                 const alloc_model::Params<double>& model_params) {
    if ((X.array() < 0).any()) {
        return "X must be nonnegative";
    }
    if (model_params.alpha.size() != static_cast<size_t>(X.rows())) {
        return "Number of alpha parameters must be equal to number of rows of "
               "X";
    }
    if (model_params.beta.size() != static_cast<size_t>(z)) {
        return "Number of beta parameters must be equal to z";
    }
    return "";
}

tensord<3>
bld::seq_greedy_bld(const matrixd& X, size_t z,
                    const alloc_model::Params<double>& model_params) {
    {
        auto error_msg = check_bld_params(X, z, model_params);
        if (!error_msg.empty()) {
            throw std::invalid_argument(error_msg);
        }
    }

    long x = X.rows();
    long y = X.cols();

    // exit early if sum is 0
    if (std::abs(X.sum()) <= std::numeric_limits<double>::epsilon()) {
        tensord<3> S(x, y, z);
        S.setZero();
        return S;
    }

    // variables to update during the iterations
    tensord<3> S(x, y, z);
    S.setZero();
    matrixd S_ipk = matrixd::Zero(x, z); // S_{i+k}
    vectord S_ppk = vectord::Zero(z);    // S_{++k}
    matrixd S_pjk = matrixd::Zero(y, z); // S_{+jk}
    double sum_alpha = std::accumulate(model_params.alpha.begin(),
                                       model_params.alpha.end(), 0.0);

    // sample X matrix one by one
    auto matrix_sampler = util::sample_ones_noreplace(X);
    int ii, jj;
    for (const auto& sample : matrix_sampler) {
        std::tie(ii, jj) = sample;

        // find the bin that increases log marginal the largest when it is
        // incremented
        size_t kmax = 0;
        double curr_max = std::numeric_limits<double>::lowest();
        for (size_t k = 0; k < z; ++k) {
            double log_marginal_change =
                std::log(model_params.alpha[ii] + S_ipk(ii, k)) -
                std::log(sum_alpha + S_ppk(k));
            log_marginal_change +=
                -(std::log(1 + S(ii, jj, k)) -
                  std::log(model_params.beta[k] + S_pjk(jj, k)));

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

std::tuple<matrixd, matrixd, vectord>
bld::bld_fact(const tensord<3>& S,
              const alloc_model::Params<double>& model_params, double eps) {
    auto x = static_cast<size_t>(S.dimension(0));
    auto y = static_cast<size_t>(S.dimension(1));
    auto z = static_cast<size_t>(S.dimension(2));

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

    matrixd W(x, z);
    matrixd H(z, y);
    vectord L(y);

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
    vectord W_colsum = W.colwise().sum() + vectord::Constant(W.cols(), eps);
    vectord H_colsum = H.colwise().sum() + vectord::Constant(H.cols(), eps);
    W = W.array().rowwise() / W_colsum.array();
    H = H.array().rowwise() / H_colsum.array();

    return std::make_tuple(W, H, L);
}

tensord<3> bld::bld_mult(const matrixd& X, size_t z,
                         const alloc_model::Params<double>& model_params,
                         size_t max_iter, bool use_psi_appr, double eps) {
    {
        auto error_msg = check_bld_params(X, z, model_params);
        if (!error_msg.empty()) {
            throw std::invalid_argument(error_msg);
        }
    }

#ifdef USE_OPENMP
    const int num_threads = std::thread::hardware_concurrency();
#endif

    auto x = static_cast<size_t>(X.rows());
    auto y = static_cast<size_t>(X.cols());

    // initialize tensor S
    tensord<3> S(x, y, z);
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
    cuda::HostMemory2D<const double> X_host(X.data(), x, y);
    cuda::DeviceMemory2D<double> X_device(x, y);
    cuda::copy2D(X_device, X_host, cudaMemcpyHostToDevice);
#endif

    tensord<3> grad_plus(x, y, z);
    matrixd grad_minus(x, z);
    matrixd nom_mult(x, y);
    matrixd denom_mult(x, y);
    matrixd alpha_eph(x, z);
    vectord alpha_eph_sum(z);
    matrixd beta_eph(y, z);
    tensord<2> S_pjk_tensor(y, z);
    tensord<2> S_ipk_tensor(x, z);
    tensord<2> S_ijp_tensor(x, y);

    Eigen::Map<const vectord> alpha(model_params.alpha.data(),
                                    model_params.alpha.size());
    Eigen::Map<const vectord> beta(model_params.beta.data(),
                                   model_params.beta.size());

#ifdef USE_OPENMP
    Eigen::ThreadPool tp(num_threads);
    Eigen::ThreadPoolDevice thread_dev(&tp, num_threads);
#endif

    const auto psi_fn = use_psi_appr ? util::psi_appr<double> : gsl_sf_psi;
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

        Eigen::Map<matrixd> S_pjk(S_pjk_tensor.data(), y, z);
        Eigen::Map<matrixd> S_ipk(S_ipk_tensor.data(), x, z);
        Eigen::Map<matrixd> S_ijp(S_ijp_tensor.data(), x, y);

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
                double xdiv = 1.0 / (X(i, j) + eps);
                double nom_sum = 0, denom_sum = 0;
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
                double s_ij = S_ijp(i, j);
                double x_over_s = X(i, j) / (s_ij + eps);
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
 * @brief Calculate step size to use during gradient ascent iterations.
 *
 * @param step Current step.
 *
 * @return Step size to use at step n.
 */
static double eta(size_t step) { return 0.1 / std::pow(step + 1, 0.55); }

tensord<3> bld::bld_add(const matrixd& X, size_t z,
                        const alloc_model::Params<double>& model_params,
                        size_t max_iter, double eps) {
    {
        auto error_msg = check_bld_params(X, z, model_params);
        if (!error_msg.empty()) {
            throw std::invalid_argument(error_msg);
        }
    }

    long x = X.rows(), y = X.cols();
    tensord<3> S(x, y, z);
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

    tensord<2> S_pjk(y, z); // S_{+jk}
    tensord<2> S_ipk(x, z); // S_{i+k}
    matrixd alpha_eph(x, z);
    matrixd beta_eph(y, z);
    tensord<3> grad_S(x, y, z);
    tensord<3> eps_tensor(x, y, z);
    eps_tensor.setConstant(eps);
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
        vectord alpha_eph_sum = alpha_eph.colwise().sum();
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
        tensord<3> Z = S.log() + eps_tensor;
        tensord<2> S_mult = (S * grad_S).sum(shape<1>({2}));
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

details::CollapsedGibbsComputer::CollapsedGibbsComputer(
    const matrixd& X, size_t z, const alloc_model::Params<double>& model_params,
    size_t max_iter, double eps)
    : model_params(model_params),
      one_sampler_repl(util::sample_ones_replace(X, max_iter)),
      one_sampler_no_repl(util::sample_ones_noreplace(X)),
      U_ipk(matrixd::Zero(X.rows(), z)), U_ppk(vectord::Zero(z)),
      U_pjk(matrixd::Zero(X.cols(), z)),
      sum_alpha(std::accumulate(model_params.alpha.begin(),
                                model_params.alpha.end(), 0.0)),
      eps(eps), rnd_gen(util::gsl_rng_wrapper(gsl_rng_alloc(gsl_rng_taus),
                                              gsl_rng_free)) {}

void details::CollapsedGibbsComputer::increment_sampling(size_t i, size_t j,
                                                         tensord<3>& S_prev) {
    vectord prob(U_ppk.cols());
    Eigen::Map<vectord> beta(model_params.beta.data(), 1,
                             model_params.beta.size());
    std::vector<unsigned int> multinomial_sample(
        static_cast<unsigned long>(U_ppk.cols()));

    vectord alpha_row = U_ipk.row(i).array() + model_params.alpha[i];
    vectord beta_row = (U_pjk.row(j) + beta).array();
    vectord sum_alpha_row = U_ppk.array() + sum_alpha;
    prob = alpha_row.array() * beta_row.array() / (sum_alpha_row.array() + eps);

    gsl_ran_multinomial(rnd_gen.get(), static_cast<size_t>(prob.cols()), 1,
                        prob.data(), multinomial_sample.data());
    auto k = std::distance(
        multinomial_sample.begin(),
        std::max_element(multinomial_sample.begin(), multinomial_sample.end()));

    ++S_prev(i, j, k);
    ++U_ipk(i, k);
    ++U_ppk(k);
    ++U_pjk(j, k);
}

void details::CollapsedGibbsComputer::decrement_sampling(size_t i, size_t j,
                                                         tensord<3>& S_prev) {
    vectord prob(U_ppk.cols());
    Eigen::Map<vectord> beta(model_params.beta.data(), 1,
                             model_params.beta.size());
    std::vector<unsigned int> multinomial_sample(
        static_cast<unsigned long>(U_ppk.cols()));

    // todo: can we take a fiber from S with contiguous memory?
    for (long k = 0; k < S_prev.dimension(2); ++k) {
        prob(k) = S_prev(i, j, k);
    }

    gsl_ran_multinomial(rnd_gen.get(), static_cast<size_t>(prob.cols()), 1,
                        prob.data(), multinomial_sample.data());
    auto k = std::distance(
        multinomial_sample.begin(),
        std::max_element(multinomial_sample.begin(), multinomial_sample.end()));

    --S_prev(i, j, k);
    --U_ipk(i, k);
    --U_ppk(k);
    --U_pjk(j, k);
}

void details::CollapsedGibbsComputer::operator()(size_t curr_step,
                                                 tensord<3>& S_prev) {
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

util::Generator<tensord<3>, details::CollapsedGibbsComputer>
bld::collapsed_gibbs(const matrixd& X, size_t z,
                     const alloc_model::Params<double>& model_params,
                     size_t max_iter, double eps) {
    {
        auto error_msg = check_bld_params(X, z, model_params);
        if (!error_msg.empty()) {
            throw std::invalid_argument(error_msg);
        }
    }

    tensord<3> init_val(X.rows(), X.cols(), z);
    init_val.setZero();

    util::Generator<tensord<3>, details::CollapsedGibbsComputer> gen(
        init_val, max_iter + 2,
        details::CollapsedGibbsComputer(X, z, model_params, max_iter, eps));

    ++gen.begin();

    return gen;
}

tensord<3> bld::collapsed_icm(const matrixd& X, size_t z,
                              const alloc_model::Params<double>& model_params,
                              size_t max_iter, double eps) {
    {
        auto error_msg = check_bld_params(X, z, model_params);
        if (!error_msg.empty()) {
            throw std::invalid_argument(error_msg);
        }
    }

    const auto x = static_cast<size_t>(X.rows());
    const auto y = static_cast<size_t>(X.cols());

    tensord<3> U(x, y, z);
    U.setZero();

    matrixd U_ipk = matrixd::Zero(x, z);
    vectord U_ppk = vectord::Zero(z);
    matrixd U_pjk = matrixd::Zero(y, z);

    const double sum_alpha = std::accumulate(model_params.alpha.begin(),
                                             model_params.alpha.end(), 0.0);

    vectord prob(U_ppk.cols());
    Eigen::Map<const vectord> beta(model_params.beta.data(), 1,
                                   model_params.beta.size());

    // initializing increment sampling without multinomial
    size_t i, j;
    for (const auto& pair : util::sample_ones_noreplace(X)) {
        std::tie(i, j) = pair;

        vectord alpha_row = U_ipk.row(i).array() + model_params.alpha[i];
        vectord beta_row = (U_pjk.row(j) + beta).array();
        vectord sum_alpha_row = U_ppk.array() + sum_alpha;
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
        vectord alpha_row = U_ipk.row(i).array() + model_params.alpha[i];
        vectord beta_row = (U_pjk.row(j) + beta).array();
        vectord sum_alpha_row = U_ppk.array() + sum_alpha;
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

/* ===================== bld_appr IMPLEMENTATION
 * ============================ */

static void update_X_hat(const matrixd& nu, const matrixd& mu, matrixd& X_hat) {
    X_hat = nu * mu;
}

static void update_orig_over_appr(const matrixd& X, const matrixd& X_hat,
                                  const matrixd& eps_mat,
                                  matrixd& orig_over_appr) {
    orig_over_appr = X.array() / (X_hat + eps_mat).array();
}

static void update_orig_over_appr_squared(const matrixd& X,
                                          const matrixd& X_hat,
                                          const matrixd& eps_mat,
                                          matrixd& orig_over_appr_squared) {
    matrixd appr_squared = X_hat.array().pow(2);
    orig_over_appr_squared = X.array() / (appr_squared + eps_mat).array();
}

static void update_S(const matrixd& orig_over_appr, const matrixd& nu,
                     const matrixd& mu, tensord<3>& S) {
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

static void update_alpha_eph(const vectord& alpha, const tensord<3>& S,
                             matrixd& alpha_eph) {
    auto x = static_cast<size_t>(S.dimension(0));
    auto z = static_cast<size_t>(S.dimension(2));

    tensord<2> S_ipk_tensor = S.sum(shape<1>({1}));
    Eigen::Map<matrixd> S_ipk(S_ipk_tensor.data(), x, z);
    alpha_eph = S_ipk.array().colwise() + alpha.transpose().array();
}

static void update_beta_eph(const vectord& beta, const tensord<3>& S,
                            matrixd& beta_eph) {
    auto y = static_cast<size_t>(S.dimension(1));
    auto z = static_cast<size_t>(S.dimension(2));

    tensord<2> S_pjk_tensor = S.sum(shape<1>({0}));
    Eigen::Map<matrixd> S_pjk(S_pjk_tensor.data(), y, z);
    beta_eph = S_pjk.array().rowwise() + beta.array();
}

static void update_grad_plus(const matrixd& beta_eph, const tensord<3>& S,
                             tensord<3>& grad_plus) {
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

static void update_grad_minus(const matrixd& alpha_eph, matrixd& grad_minus) {
    auto x = static_cast<size_t>(grad_minus.rows());
    auto z = static_cast<size_t>(grad_minus.cols());

    vectord alpha_eph_sum = alpha_eph.colwise().sum();
    for (size_t i = 0; i < x; ++i) {
        for (size_t k = 0; k < z; ++k) {
            grad_minus(i, k) =
                gsl_sf_psi(alpha_eph_sum(k)) - gsl_sf_psi(alpha_eph(i, k));
        }
    }
}

static void update_nu(const matrixd& orig_over_appr,
                      const matrixd& orig_over_appr_squared, const matrixd& mu,
                      const tensord<3>& grad_plus, const matrixd& grad_minus,
                      double eps, matrixd& nu) {
    auto x = static_cast<size_t>(grad_plus.dimension(0));
    auto y = static_cast<size_t>(grad_plus.dimension(1));
    auto z = static_cast<size_t>(grad_plus.dimension(2));

    matrixd nom = matrixd::Zero(x, z);
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
    matrixd denom = matrixd::Zero(x, z);
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
    vectord nu_rowsum_plus_eps =
        nu.rowwise().sum() + vectord::Constant(nu.rows(), eps).transpose();
    nu = nu.array().colwise() / nu_rowsum_plus_eps.transpose().array();
}

static void update_mu(const matrixd& orig_over_appr,
                      const matrixd& orig_over_appr_squared, const matrixd& nu,
                      const tensord<3>& grad_plus, const matrixd& grad_minus,
                      double eps, matrixd& mu) {
    auto x = static_cast<size_t>(grad_plus.dimension(0));
    auto y = static_cast<size_t>(grad_plus.dimension(1));
    auto z = static_cast<size_t>(grad_plus.dimension(2));

    matrixd nom = matrixd::Zero(z, y);
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
    matrixd denom = matrixd::Zero(z, y);
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
    vectord mu_colsum_plus_eps =
        mu.colwise().sum() + vectord::Constant(mu.cols(), eps);
    mu = mu.array().rowwise() / mu_colsum_plus_eps.array();
}

std::tuple<tensord<3>, matrixd, matrixd>
bld::bld_appr(const matrixd& X, size_t z,
              const alloc_model::Params<double>& model_params, size_t max_iter,
              double eps) {
    {
        auto error_msg = check_bld_params(X, z, model_params);
        if (!error_msg.empty()) {
            throw std::invalid_argument(error_msg);
        }
    }

    auto x = static_cast<size_t>(X.rows());
    auto y = static_cast<size_t>(X.cols());

    Eigen::Map<const vectord> alpha(model_params.alpha.data(), 1,
                                    model_params.alpha.size());
    Eigen::Map<const vectord> beta(model_params.beta.data(), 1,
                                   model_params.beta.size());

    matrixd nu(x, z);
    matrixd mu(y, z);
    // initialize nu and mu using Dirichlet
    {
        util::gsl_rng_wrapper rnd_gen(gsl_rng_alloc(gsl_rng_taus), gsl_rng_free);
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

    tensord<3> grad_plus(x, y, z);
    matrixd grad_minus(x, z);
    tensord<3> S(x, y, z);
    matrixd eps_mat = matrixd::Constant(x, y, eps);
    matrixd X_hat, orig_over_appr, orig_over_appr_squared, alpha_eph, beta_eph;
    for (size_t eph = 0; eph < max_iter; ++eph) {
        // update helpers
        update_X_hat(nu, mu, X_hat);
        update_orig_over_appr(X, X_hat, eps_mat, orig_over_appr);
        update_orig_over_appr_squared(X, X_hat, eps_mat,
                                      orig_over_appr_squared);
        update_S(orig_over_appr, nu, mu, S);
        update_alpha_eph(alpha, S, alpha_eph);
        update_beta_eph(beta, S, beta_eph);
        update_grad_plus(beta_eph, S, grad_plus);
        update_grad_minus(alpha_eph, grad_minus);

        // update nu
        update_nu(orig_over_appr, orig_over_appr_squared, mu, grad_plus,
                  grad_minus, eps, nu);

        // update helpers
        update_X_hat(nu, mu, X_hat);
        update_orig_over_appr(X, X_hat, eps_mat, orig_over_appr);
        update_orig_over_appr_squared(X, X_hat, eps_mat,
                                      orig_over_appr_squared);
        update_S(orig_over_appr, nu, mu, S);
        update_alpha_eph(alpha, S, alpha_eph);
        update_beta_eph(beta, S, beta_eph);
        update_grad_plus(beta_eph, S, grad_plus);
        update_grad_minus(alpha_eph, grad_minus);

        // update mu
        update_mu(orig_over_appr, orig_over_appr_squared, nu, grad_plus,
                  grad_minus, eps, mu);
    }

    update_X_hat(nu, mu, X_hat);
    update_orig_over_appr(X, X_hat, eps_mat, orig_over_appr);
    update_S(orig_over_appr, nu, mu, S);

    return std::make_tuple(S, nu, mu);
}
