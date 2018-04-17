#pragma once

#include "alloc_model/alloc_model_funcs.hpp"
#include "alloc_model/alloc_model_params.hpp"
#include "defs.hpp"
#include "util/generator.hpp"
#include <util/sampling.hpp>
#include <util/util.hpp>

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
     * bnmf_algs::alloc_model::Params<double>.
     * @param max_iter Maximum number of iterations.
     * @param eps Floating point epsilon value to be used to prevent division by
     * 0 errors.
     */
    explicit CollapsedGibbsComputer(
        const matrixd& X, size_t z,
        const alloc_model::Params<double>& model_params, size_t max_iter,
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
    alloc_model::Params<double> model_params;
    // computation variables
  private:
    util::Generator<std::pair<int, int>, details::SampleOnesReplaceComputer<double>>
        one_sampler_repl;
    util::Generator<std::pair<int, int>, details::SampleOnesNoReplaceComputer<double>>
        one_sampler_no_repl;
    matrixd U_ipk;
    vectord U_ppk;
    matrixd U_pjk;
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
tensord<3>
seq_greedy_bld(const matrixd& X, size_t z,
               const alloc_model::Params<double>& model_params);

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
std::tuple<matrixd, matrixd, vectord>
bld_fact(const tensord<3>& S,
         const alloc_model::Params<double>& model_params,
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
 * @param use_psi_appr If true, use util::psi_appr function to compute the digamma
 * function approximately. If false, compute the digamma function exactly.
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
tensord<3> bld_mult(const matrixd& X, size_t z,
                    const alloc_model::Params<double>& model_params,
                    size_t max_iter = 1000, bool use_psi_appr = false,
                    double eps = 1e-50);

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
tensord<3> bld_add(const matrixd& X, size_t z,
                   const alloc_model::Params<double>& model_params,
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
util::Generator<tensord<3>, details::CollapsedGibbsComputer>
collapsed_gibbs(const matrixd& X, size_t z,
                const alloc_model::Params<double>& model_params,
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
tensord<3> collapsed_icm(const matrixd& X, size_t z,
                         const alloc_model::Params<double>& model_params,
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
std::tuple<bnmf_algs::tensord<3>, bnmf_algs::matrixd, bnmf_algs::matrixd>
bld_appr(const matrixd& X, size_t z,
         const alloc_model::Params<double>& model_params,
         size_t max_iter = 1000, double eps = 1e-50);
} // namespace bld
} // namespace bnmf_algs
