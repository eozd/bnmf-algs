#pragma once

#include "alloc_model/alloc_model_params.hpp"
#include "defs.hpp"
#include "util/sampling.hpp"
#include "util_details.hpp"

namespace bnmf_algs {
namespace details {
/**
 * @brief Computer type that will compute the next collapsed gibbs sample when
 * its operator() is invoked.
 *
 * CollapsedGibbsComputer will compute a new bnmf_algs::tensord<3> object from
 * the previous sample when its operator() is invoked.
 *
 * @tparam T Type of the matrix/tensor entries.
 * @tparam Scalar Type of the scalars used as model parameters.
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
} // namespace details

namespace bld {
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
 * @tparam T Type of the matrix/tensor entries.
 * @tparam Scalar Type of the scalars used as model parameters.
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
} // namespace bld
} // namespace bnmf_algs
