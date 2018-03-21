#pragma once

#include <util/util.hpp>
#include <util/sampling.hpp>
#include "allocation_model/alloc_model_params.hpp"
#include "allocation_model/alloc_model_funcs.hpp"
#include "defs.hpp"
#include "util/generator.hpp"
#include "util/wrappers.hpp"

namespace bnmf_algs {
namespace details {
/**
 * @brief Computer type that will compute the next collapsed gibbs sample when
 * its operator() is invoked.
 *
 * CollapsedGibbsComputer will compute a new bnmf_algs::tensord<3> object from
 * the previous sample when its operator() is invoked.
 */
class CollapsedGibbsComputer {
  public:
    /**
     * @brief Construct a new CollapsedGibbsComputer.
     *
     * @param X Matrix \f$X\f$ of size \f$x \times y\f$ that will be used during
     * sampling.
     * @param z Depth of the output tensor \f$S\f$ with size \f$x \times y
     * \times z\f$.
     * @param model_params Allocation model parameters. See
     * bnmf_algs::allocation_model::AllocModelParams.
     * @param max_iter Maximum number of iterations.
     * @param eps Floating point epsilon value to be used to prevent division by
     * 0 errors.
     */
    explicit CollapsedGibbsComputer(
        const matrix_t& X, size_t z,
        const allocation_model::AllocModelParams& model_params, size_t max_iter,
        double eps);

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
    void operator()(size_t curr_step, tensord<3>& S_prev);

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
    void increment_sampling(size_t i, size_t j, tensord<3>& S_prev);
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
    void decrement_sampling(size_t i, size_t j, tensord<3>& S_prev);

  private:
    allocation_model::AllocModelParams model_params;
    // computation variables
  private:
    util::Generator<std::pair<int, int>, details::SampleOnesComputer>
        one_sampler_repl;
    util::Generator<std::pair<int, int>, details::SampleOnesComputer>
        one_sampler_no_repl;
    matrix_t U_ipk;
    vector_t U_ppk;
    matrix_t U_pjk;
    double sum_alpha;
    double eps;
    util::gsl_rng_wrapper rnd_gen;
};
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
 * \todo Explain sequential greedy algorithm. (first need to understand)
 *
 * @param X Nonnegative matrix of size \f$x \times y\f$ to decompose.
 * @param z Number of matrices into which matrix \f$X\f$ will be decomposed.
 * This is the depth of the output tensor \f$S\f$.
 * @param model_params Allocation model parameters. See
 * bnmf_algs::allocation_model::AllocModelParams.
 *
 * @return Tensor \f$S\f$ of size \f$x \times y \times z\f$ where \f$X =
 * S_{ij+}\f$.
 *
 * @throws std::invalid_argument if matrix \f$X\f$ is not nonnegative, if size
 * of alpha is not \f$x\f$, if size of beta is not \f$z\f$.
 */
tensord<3>
seq_greedy_bld(const matrix_t& X, size_t z,
               const allocation_model::AllocModelParams& model_params);

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
 * bnmf_algs::allocation_model::AllocModelParams.
 * @param eps Floating point epsilon value to be used to prevent division by 0
 * errors.
 *
 * @return std::tuple of matrices \f$W_{x \times z}, H_{z \times y}\f$ and
 * vector \f$L_y\f$.
 */
std::tuple<matrix_t, matrix_t, vector_t>
bld_fact(const tensord<3>& S,
         const allocation_model::AllocModelParams& model_params,
         double eps = 1e-50);

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
 * \todo Explain multiplicative algorithm (first need to understand)
 *
 * @param X Nonnegative matrix of size \f$x \times y\f$ to decompose.
 * @param z Number of matrices into which matrix \f$X\f$ will be decomposed.
 * This is the depth of the output tensor \f$S\f$.
 * @param model_params Allocation model parameters. See
 * bnmf_algs::allocation_model::AllocModelParams.
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
tensord<3> bld_mult(const matrix_t& X, size_t z,
                    const allocation_model::AllocModelParams& model_params,
                    size_t max_iter = 1000, double eps = 1e-50);

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
 * \todo Explain additive gradient ascent algorithm (first need to understand)
 *
 * @param X Nonnegative matrix of size \f$x \times y\f$ to decompose.
 * @param z Number of matrices into which matrix \f$X\f$ will be decomposed.
 * This is the depth of the output tensor \f$S\f$.
 * @param model_params Allocation model parameters. See
 * bnmf_algs::allocation_model::AllocModelParams.
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
tensord<3> bld_add(const matrix_t& X, size_t z,
                   const allocation_model::AllocModelParams& model_params,
                   size_t max_iter = 1000, double eps = 1e-50);

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
 * bnmf_algs::allocation_model::AllocModelParams.
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
util::Generator<tensord<3>, details::CollapsedGibbsComputer>
collapsed_gibbs(const matrix_t& X, size_t z,
                const allocation_model::AllocModelParams& model_params,
                size_t max_iter = 1000, double eps = 1e-50);

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
 * bnmf_algs::allocation_model::AllocModelParams.
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
tensord<3> collapsed_icm(const matrix_t& X, size_t z,
                         const allocation_model::AllocModelParams& model_params,
                         size_t max_iter = 1000, double eps = 1e-50);

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
 * bnmf_algs::allocation_model::AllocModelParams.
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
std::tuple<bnmf_algs::tensord<3>, bnmf_algs::matrix_t, bnmf_algs::matrix_t>
bld_appr(const matrix_t& X, size_t z,
         const allocation_model::AllocModelParams& model_params,
         size_t max_iter = 1000, double eps = 1e-50);
} // namespace bld
} // namespace bnmf_algs
