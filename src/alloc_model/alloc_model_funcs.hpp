#pragma once

#include "alloc_model/alloc_model_params.hpp"
#include "defs.hpp"
#include "util/generator.hpp"
#include "util/util.hpp"
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>
#include <tuple>
#include <vector>

namespace bnmf_algs {
namespace alloc_model {
// forward declaration needed to use this in details functions
template <typename T, typename Scalar>
double log_marginal_S(const tensor_t<T, 3>& S,
                      const Params<Scalar>& model_params);
} // namespace alloc_model

namespace details {

/**
 * @brief Compute the first term of the sum when calculating log marginal of S.
 *
 * @tparam Scalar Type of the model parameters.
 * @tparam T Value type of the tensor S.
 * @param S Tensor S.
 * @param alpha Parameter vector of Dirichlet prior for matrix \f$W\f$ of size
 * \f$x\f$.
 *
 * @return Value of the first term
 */
template <typename T, typename Scalar>
double compute_first_term(const tensor_t<T, 3>& S,
                          const std::vector<Scalar>& alpha) {
    const long x = S.dimension(0);
    const long y = S.dimension(1);
    const long z = S.dimension(2);

    // loggamma(sum(alpha))
    const double log_gamma_sum =
        gsl_sf_lngamma(std::accumulate(alpha.begin(), alpha.end(), 0.0));

    // sum(loggamma(alpha))
    std::vector<double> log_gamma_alpha(alpha.size());
    std::transform(
        alpha.begin(), alpha.end(), log_gamma_alpha.begin(),
        [](const Scalar alpha_i) { return gsl_sf_lngamma(alpha_i); });

    const double sum_log_gamma =
        std::accumulate(log_gamma_alpha.begin(), log_gamma_alpha.end(), 0.0);

    double first = 0;
    #pragma omp parallel for reduction(+:first)
    for (int k = 0; k < z; ++k) {
        double sum = 0;
        for (int i = 0; i < x; ++i) {
            sum += alpha[i];
            for (int j = 0; j < y; ++j) {
                sum += S(i, j, k);
            }
        }
        first -= gsl_sf_lngamma(sum);

        for (int i = 0; i < x; ++i) {
            sum = alpha[i];
            for (int j = 0; j < y; ++j) {
                sum += S(i, j, k);
            }
            first += gsl_sf_lngamma(sum);
        }
    }

    double base = log_gamma_sum * z - sum_log_gamma * z;

    return base + first;
}

/**
 * @brief Compute the second term of the sum when calculating log marginal of S.
 *
 * @tparam Scalar Type of the model parameters.
 * @tparam T Value type of tensor S.
 * @param S Tensor S.
 * @param beta Parameter vector of Dirichlet prior for matrix \f$H\f$ of size
 * \f$z\f$.
 *
 * @return Value of the second term
 */
template <typename T, typename Scalar>
double compute_second_term(const tensor_t<T, 3>& S,
                           const std::vector<Scalar>& beta) {
    const long x = S.dimension(0);
    const long y = S.dimension(1);
    const long z = S.dimension(2);

    // loggamma(sum(beta))
    const double log_gamma_sum =
        gsl_sf_lngamma(std::accumulate(beta.begin(), beta.end(), 0.0));

    // sum(loggamma(beta))
    std::vector<double> log_gamma_beta(beta.size());
    std::transform(beta.begin(), beta.end(), log_gamma_beta.begin(),
                   [](const Scalar beta) { return gsl_sf_lngamma(beta); });

    const double sum_log_gamma =
        std::accumulate(log_gamma_beta.begin(), log_gamma_beta.end(), 0.0);

    double second = 0;
    #pragma omp parallel for reduction(+:second)
    for (int j = 0; j < y; ++j) {
        double sum = 0;
        for (int k = 0; k < z; ++k) {
            sum += beta[k];
            for (int i = 0; i < x; ++i) {
                sum += S(i, j, k);
            }
        }
        second -= gsl_sf_lngamma(sum);

        for (int k = 0; k < z; ++k) {
            sum = beta[k];
            for (int i = 0; i < x; ++i) {
                sum += S(i, j, k);
            }
            second += gsl_sf_lngamma(sum);
        }
    }

    double base = log_gamma_sum * y - sum_log_gamma * y;
    return base + second;
}

/**
 * @brief Compute the third term of the sum when calculating log marginal of S.
 *
 * @tparam Scalar Type of the model parameters.
 * @tparam T Value type of tensor S.
 * @param S Tensor S.
 * @param a Shape parameter of Gamma distribution.
 * @param b Rate parameter of Gamma distribution.
 *
 * @return Value of the third term
 */
template <typename T, typename Scalar>
double compute_third_term(const tensor_t<T, 3>& S, Scalar a, Scalar b) {
    const long x = S.dimension(0);
    const long y = S.dimension(1);
    const long z = S.dimension(2);

    const double log_gamma = -gsl_sf_lngamma(a);
    const double a_log_b = a * std::log(b);

    double third = 0;
    #pragma omp parallel for reduction(+:third)
    for (int j = 0; j < y; ++j) {
        double sum = a;
        for (int i = 0; i < x; ++i) {
            for (int k = 0; k < z; ++k) {
                sum += S(i, j, k);
            }
        }
        third += gsl_sf_lngamma(sum);
        third -= sum * std::log(b + 1);
    }

    double base = log_gamma * y + a_log_b * y;
    return base + third;
}

/**
 * @brief Compute the fourth term of the sum when calculating log marginal of S.
 *
 * @tparam T Value type of tensor S.
 * @param S Tensor S.
 *
 * @return Value of the fourth term
 */
template <typename T> double compute_fourth_term(const tensor_t<T, 3>& S) {
    const long x = S.dimension(0);
    const long y = S.dimension(1);
    const long z = S.dimension(2);

    double fourth = 0;
    #pragma omp parallel for reduction(+:fourth)
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            for (int k = 0; k < z; ++k) {
                fourth += gsl_sf_lngamma(S(i, j, k) + 1);
            }
        }
    }
    return fourth;
}

/**
 * @brief Class to compute total \f$p(X|\Theta) = \sum_S p(X|S)p(S|\Theta)\f$
 * for all possible allocation tensors while storing shared state between
 * function invocations.
 *
 * @tparam Integer Integer types used in matrices/tensors.
 * @tparam Scalar Model parameter types.
 */
template <typename Integer, typename Scalar> class TotalMarginalCalculator {
  public:
    /**
     * @brief Construct the calculator class and initialize computation
     * variables.
     *
     * @param X Matrix to allocate.
     * @param model_params Model parameters.
     */
    TotalMarginalCalculator(const matrix_t<Integer>& X,
                            const alloc_model::Params<Scalar>& model_params)
        : X(X), model_params(model_params),
          S(X.rows(), X.cols(), model_params.beta.size()) {
        // initialize variables to use during recursive computation

        // get all partitions for every nonzero element of X
        const Integer z = model_params.beta.size();
        for (long i = 0; i < X.rows(); ++i) {
            for (long j = 0; j < X.cols(); ++j) {
                if (X(i, j) != 0) {
                    ii.push_back(i);
                    jj.push_back(j);
                    alloc_vec.push_back(
                        util::partition_change_indices(X(i, j), z));
                }
            }
        }

        // initialize allocation tensors
        S.setZero();
        for (long i = 0; i < X.rows(); ++i) {
            for (long j = 0; j < X.cols(); ++j) {
                S(i, j, 0) = X(i, j);
            }
        }
        S_pjk = S.sum(shape<1>({0}));
        S_ipk = S.sum(shape<1>({1}));
        S_ppk = S.sum(shape<2>({0, 1}));

        sum_alpha = std::accumulate(model_params.alpha.begin(),
                                    model_params.alpha.end(), Scalar());
    }

    /**
     * @brief Calculate total marginal by calculating
     * \f$\sum_S p(X|S)p(S|\Theta)\f$ for every possible allocation tensor
     * \f$S\f$.
     *
     * @return Total marginal, \f$p(X|\Theta)\f$.
     */
    double calc_marginal() {
        const double init_log_marginal =
            alloc_model::log_marginal_S(S, model_params);

        return marginal_recursive(0, init_log_marginal);
    }

  private:
    /**
     * @brief Calculate the difference in \f$\log{P(S)}\f$ value when element at
     * tensor bin (i, j, k) is incremented by 1.
     *
     * @param i First index of the tensor.
     * @param j Second index of the tensor.
     * @param k Third index of the tensor.
     * @return Change in log marginal when (i, j, k) element is incremented
     * by 1.
     */
    double log_marginal_change_on_increment(size_t i, size_t j, size_t k) {
        return std::log(model_params.alpha[i] + S_ipk(i, k)) -
               std::log(sum_alpha + S_ppk(k)) - std::log(1 + S(i, j, k)) +
               std::log(model_params.beta[k] + S_pjk(j, k));
    }

    /**
     * @brief Calculate the difference in \f$\log{P(S)}\f$ value when element at
     * tensor bin (i, j, k) is decremented by 1.
     *
     * @param i First index of the tensor.
     * @param j Second index of the tensor.
     * @param k Third index of the tensor.
     * @return Change in log marginal when (i, j, k) element is decremented
     * by 1.
     */
    double log_marginal_change_on_decrement(size_t i, size_t j, size_t k) {
        double result = -(std::log(model_params.alpha[i] + S_ipk(i, k) - 1) -
                          std::log(sum_alpha + S_ppk(k) - 1) -
                          std::log(1 + S(i, j, k) - 1) +
                          std::log(model_params.beta[k] + S_pjk(j, k) - 1));

        return result;
    }

    /**
     * @brief Recursively enumerate and sum \f$\log{P(S_i)}\f$ values for all
     * possible allocation tensors \f$S_i\f$.
     *
     * This function recursively enumerates \f$\log{P(S_i)}\f$ values for all
     * possible allocation tensors by trying all possible k length partitions
     * of every nonzero element of \f$X\f$. Here, k is the rank of the
     * decomposition (depth of tensor \f$S\f$). Log marginal values are
     * computed using the "change in logPS" formulas, and are not computed from
     * scratch for every possible \f$S\f$.
     *
     * @param fiber_index Index of the nonzero fiber of \f$S\f$ to allocate in
     * this function invocation. Every possible partition of this fiber will be
     * tried.
     * @param prev_log_marginal Previous log marginal value to be used when
     * calculating the new log marginal values by finding the difference.
     *
     * @return Total log marginal value calculated by trying every possible
     * allocation tensor enumerated using all possible allocations of every
     * fiber with index greater than or equal to the given index.
     */
    double marginal_recursive(const size_t fiber_index,
                              double prev_log_marginal) {

        // get current fiber partitions and indices
        const auto& part_changes = alloc_vec[fiber_index];
        const size_t i = ii[fiber_index];
        const size_t j = jj[fiber_index];

        const Integer old_value = S(i, j, 0);

        // if true, then base case; else, recursive part.
        const bool last_fiber = fiber_index == (ii.size() - 1);

        double result =
            last_fiber ? std::exp(prev_log_marginal)
                       : marginal_recursive(fiber_index + 1, prev_log_marginal);

        // calculate log marginal of every possible tensor and accumulate
        // results
        size_t incr_idx, decr_idx;
        double increment_change, decrement_change, new_log_marginal;
        for (const auto& change_idx : part_changes) {
            // indices of elements to decrement and increment
            std::tie(decr_idx, incr_idx) = change_idx;

            decrement_change = log_marginal_change_on_decrement(i, j, decr_idx);

            // modify elements
            --S(i, j, decr_idx);
            --S_pjk(j, decr_idx);
            --S_ipk(i, decr_idx);
            --S_ppk(decr_idx);

            increment_change = log_marginal_change_on_increment(i, j, incr_idx);

            // modify elements
            ++S(i, j, incr_idx);
            ++S_pjk(j, incr_idx);
            ++S_ipk(i, incr_idx);
            ++S_ppk(incr_idx);

            new_log_marginal =
                prev_log_marginal + increment_change + decrement_change;

            // accumulate
            if (last_fiber) {
                result += std::exp(new_log_marginal);
            } else {
                result += marginal_recursive(fiber_index + 1, new_log_marginal);
            }

            prev_log_marginal = new_log_marginal;
        }

        // make S same as before
        const auto last_index = S.dimension(2) - 1;
        S(i, j, last_index) = 0;
        S(i, j, 0) = old_value;
        S_pjk(j, last_index) -= old_value;
        S_pjk(j, 0) += old_value;
        S_ipk(i, last_index) -= old_value;
        S_ipk(i, 0) += old_value;
        S_ppk(last_index) -= old_value;
        S_ppk(0) += old_value;

        return result;
    }

  private:
    /**
     * @brief Matrix to try all possible allocations.
     */
    const matrix_t<Integer>& X;
    /**
     * @brief Model parameters.
     */
    const alloc_model::Params<Scalar>& model_params;
    /**
     * @brief Allocation tensor variable that will hold different possible
     * allocations.
     */
    tensor_t<Integer, 3> S;
    /**
     * @brief \f$S_{+jk}\f$.
     */
    tensor_t<Integer, 2> S_pjk;
    /**
     * @brief \f$S_{i+k}\f$.
     */
    tensor_t<Integer, 2> S_ipk;
    /**
     * @brief \f$S_{++k}\f$.
     */
    tensor_t<Integer, 1> S_ppk;
    /**
     * @brief Sum of alpha parameters. (\f$sum_i \alpha_i\f$).
     */
    Scalar sum_alpha;
    /**
     * @brief Row indices of nonzero entries of X.
     */
    std::vector<size_t> ii;
    /**
     * @brief Column indices of nonzero entries of X.
     */
    std::vector<size_t> jj;
    /**
     * @brief std::vector holding every possible k length partitions of every
     * nonzero element of X where k is the decomposition rank (depth of tensor
     * S).
     *
     * The ordering of elements in alloc_vec is the same as vectors ii and jj.
     */
    std::vector<std::vector<std::pair<size_t, size_t>>> alloc_vec;
};
} // namespace details

/**
 * @brief Namespace that contains functions related to Allocation Model \cite
 * kurutmazbayesian.
 */
namespace alloc_model {

/**
 * @brief Return prior matrices W, H and vector L according to the Bayesian
 * NMF allocation model using distribution parameters.
 *
 * According to the Bayesian NMF allocation model \cite kurutmazbayesian,
 *
 * \f$L_j \sim \mathcal{G}(a, b) \qquad W_{:k} \sim \mathcal{D}(\alpha)
 * \qquad H_{:j} \sim \mathcal{D}(\beta)\f$.
 *
 * @tparam T Value type of the matrix/vector objects.
 * @tparam Scalar Type of the model parameters.
 * @param tensor_shape Shape of the sample tensor \f$x \times y \times z\f$.
 * @param model_params Allocation Model parameters. See
 * bnmf_algs::alloc_model::Params<double>.
 *
 * @return std::tuple of \f$<W_{x \times z}, H_{z \times y}, L_y>\f$.
 *
 * @throws assertion error if number of alpha parameters is not \f$x\f$,
 * if number of beta parameters is not \f$z\f$.
 */
template <typename T, typename Scalar>
std::tuple<matrix_t<T>, matrix_t<T>, vector_t<T>>
bnmf_priors(const shape<3>& tensor_shape, const Params<Scalar>& model_params) {
    size_t x = tensor_shape[0], y = tensor_shape[1], z = tensor_shape[2];

    BNMF_ASSERT(model_params.alpha.size() == x,
                "Number of dirichlet parameters alpha must be equal to x");
    BNMF_ASSERT(model_params.beta.size() == z,
                "Number of dirichlet parameters beta must be equal to z");

    util::gsl_rng_wrapper rand_gen(gsl_rng_alloc(gsl_rng_taus), gsl_rng_free);

    // generate prior_L
    vector_t<T> prior_L(y);
    for (size_t i = 0; i < y; ++i) {
        prior_L(i) =
            gsl_ran_gamma(rand_gen.get(), model_params.a, model_params.b);
    }

    // generate prior_W
    matrix_t<T> prior_W(x, z);
    vector_t<T> dirichlet_variates(x);
    for (size_t i = 0; i < z; ++i) {
        gsl_ran_dirichlet(rand_gen.get(), x, model_params.alpha.data(),
                          dirichlet_variates.data());

        for (size_t j = 0; j < x; ++j) {
            prior_W(j, i) = dirichlet_variates(j);
        }
    }

    // generate prior_H
    matrix_t<T> prior_H(z, y);
    dirichlet_variates = vector_t<T>(z);
    for (size_t i = 0; i < y; ++i) {
        gsl_ran_dirichlet(rand_gen.get(), z, model_params.beta.data(),
                          dirichlet_variates.data());

        for (size_t j = 0; j < z; ++j) {
            prior_H(j, i) = dirichlet_variates(j);
        }
    }

    return std::make_tuple(prior_W, prior_H, prior_L);
}

/**
 * @brief Sample a tensor S from generative Bayesian NMF model using the
 * given priors.
 *
 * According to the generative Bayesian NMF model \cite kurutmazbayesian,
 *
 * \f$S_{ijk} \sim \mathcal{PO}(W_{ik}H_{kj}L_{j})\f$.
 *
 * @tparam T Value type of the matrix/vector/tensor objects.
 * @param prior_W Prior matrix for \f$W\f$ of shape \f$x \times z\f$.
 * @param prior_H Prior matrix for \f$H\f$ of shape \f$z \times y\f$.
 * @param prior_L Prior vector for \f$L\f$ of size \f$y\f$.
 *
 * @return A sample tensor of size \f$x \times y \times z\f$ from
 * generative Bayesian NMF model.
 *
 * @throws assertion error if number of columns of prior_W is not equal to
 * number of rows of prior_H, if number of columns of prior_H is not equal to
 * number of columns of prior_L
 */
template <typename T>
tensor_t<T, 3> sample_S(const matrix_t<T>& prior_W, const matrix_t<T>& prior_H,
                        const vector_t<T>& prior_L) {
    auto x = static_cast<size_t>(prior_W.rows());
    auto y = static_cast<size_t>(prior_L.cols());
    auto z = static_cast<size_t>(prior_H.rows());

    BNMF_ASSERT(prior_W.cols() == prior_H.rows(),
                "Number of columns of W is different than number of rows of H");
    BNMF_ASSERT(prior_H.cols() == prior_L.cols(),
                "Number of columns of H is different than size of L");

    // TODO: Is it OK to construct this rng object every time?
    util::gsl_rng_wrapper rand_gen(gsl_rng_alloc(gsl_rng_taus), gsl_rng_free);
    tensor_t<T, 3> sample(x, y, z);
    double mu;
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            for (size_t k = 0; k < z; ++k) {
                mu = prior_W(i, k) * prior_H(k, j) * prior_L(j);
                sample(i, j, k) = gsl_ran_poisson(rand_gen.get(), mu);
            }
        }
    }
    return sample;
}

/**
 * @brief Compute the log marginal of tensor S with respect to the given
 * distribution parameters.
 *
 * According to the Bayesian NMF allocation model \cite kurutmazbayesian,
 *
 * \f$\log{p(S)} = \log{p(W, H, L, S)} - \log{p(W, H, L | S)}\f$.
 *
 * @tparam T value type of tensor S.
 * @tparam Scalar Type of the model parameters.
 * @param S Tensor \f$S\f$ of size \f$x \times y \times z \f$.
 * @param model_params Allocation model parameters. See
 * bnmf_algs::alloc_model::Params<double>.
 *
 * @return log marginal \f$\log{p(S)}\f$.
 *
 * @throws assertion error if number of alpha parameters is not equal to
 * S.dimension(0), if number of beta parameters is not equal to S.dimension(2).
 */
template <typename T, typename Scalar>
double log_marginal_S(const tensor_t<T, 3>& S,
                      const Params<Scalar>& model_params) {
    BNMF_ASSERT(model_params.alpha.size() ==
                    static_cast<size_t>(S.dimension(0)),
                "Number of alpha parameters must be equal to S.dimension(0)");
    BNMF_ASSERT(model_params.beta.size() == static_cast<size_t>(S.dimension(2)),
                "Number of alpha parameters must be equal to S.dimension(2)");

    return details::compute_first_term(S, model_params.alpha) +
           details::compute_second_term(S, model_params.beta) +
           details::compute_third_term(S, model_params.a, model_params.b) -
           details::compute_fourth_term(S);
}

/**
 * @brief Calculate total marginal value, \f$p(X|\Theta) = \sum_S
 * p(X|S)p(S|\Theta)\f$ where \f$Theta\f$ is model parameters.
 *
 * @tparam Integer Type of the matrix entries (must be an integer type).
 * @tparam Scalar Type of the model parameters.
 * @param X Integer matrix whose every possible tensor allocation will be tried.
 * @param model_params Model parameters.
 *
 * @return Total marginal value \f$p(X|\Theta)\f$.
 */
template <typename Integer, typename Scalar>
double total_log_marginal(const matrix_t<Integer>& X,
                          const Params<Scalar>& model_params) {
    BNMF_ASSERT((X.array() >= 0).all(),
                "X must be nonnegative in alloc_model::total_log_marginal");
    BNMF_ASSERT(static_cast<size_t>(X.rows()) == model_params.alpha.size(),
                "Model parameters are incompatible with given matrix X in "
                "alloc_model::total_log_marginal");

    details::TotalMarginalCalculator<Integer, Scalar> calc(X, model_params);
    double marginal = calc.calc_marginal();

    return std::log(marginal);
}

} // namespace alloc_model
} // namespace bnmf_algs
