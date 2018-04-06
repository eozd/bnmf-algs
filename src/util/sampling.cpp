#include "util/sampling.hpp"

using namespace bnmf_algs;

details::SampleOnesComputer::SampleOnesComputer(const matrix_t& X,
                                                bool replacement)
    : replacement(replacement), vals_indices(),
      no_repl_comp([](const val_index& left, const val_index& right) {
          return left.first < right.first;
      }),
      cum_prob(), X_cols(X.cols()), X_sum(X.array().sum()),
      rnd_gen(util::make_gsl_rng(gsl_rng_taus)) {

    if (replacement) {
        cum_prob = vector_t(X.rows() * X.cols());
        auto X_arr = X.array();
        cum_prob(0) = X_arr(0);
        for (int i = 1; i < X_arr.size(); ++i) {
            cum_prob(i) = cum_prob(i - 1) + X_arr(i);
        }
    } else {
        for (int i = 0; i < X.rows(); ++i) {
            for (int j = 0; j < X.cols(); ++j) {
                double val = X(i, j);
                if (val > 0) {
                    vals_indices.emplace_back(val, std::make_pair(i, j));
                }
            }
        }
        std::make_heap(vals_indices.begin(), vals_indices.end(), no_repl_comp);
    }
}

void details::SampleOnesComputer::operator()(size_t curr_step,
                                             std::pair<int, int>& prev_val) {
    if (replacement) {
        // inverse CDF sampling
        double* begin = cum_prob.data();
        double* end = cum_prob.data() + cum_prob.cols();
        auto it = util::choice(begin, end, rnd_gen);
        long m = it - begin;

        prev_val.first = static_cast<int>(m / X_cols);
        prev_val.second = static_cast<int>(m % X_cols);
    } else {
        // max heap sampling (deterministic)
        std::pop_heap(vals_indices.begin(), vals_indices.end(), no_repl_comp);
        auto& curr_sample = vals_indices.back();
        int ii, jj;
        std::tie(ii, jj) = curr_sample.second;

        // decrement sample's value
        --curr_sample.first;
        if (curr_sample.first <= 0) {
            vals_indices.pop_back();
        }
        std::push_heap(vals_indices.begin(), vals_indices.end(), no_repl_comp);

        // store results
        prev_val.first = ii;
        prev_val.second = jj;
    }
}

util::Generator<std::pair<int, int>, details::SampleOnesComputer>
util::sample_ones(const matrix_t& X, bool replacement, size_t n) {
    if (X.array().size() == 0) {
        throw std::invalid_argument("Matrix X must have at least one element");
    }
    if ((X.array() < 0).any()) {
        throw std::invalid_argument("Matrix X must be nonnegative");
    }
    if ((X.array().abs() <= std::numeric_limits<double>::epsilon()).all()) {
        throw std::invalid_argument(
            "Matrix X must have at least one nonzero element");
    }

    size_t num_samples;
    if (replacement) {
        num_samples = n;
    } else {
        num_samples = static_cast<size_t>(X.array().sum());
    }
    // don't know the initial value
    std::pair<int, int> init_val;
    util::Generator<std::pair<int, int>, details::SampleOnesComputer> gen(
        init_val, num_samples + 1, details::SampleOnesComputer(X, replacement));

    // get rid of the first empty value
    ++gen.begin();

    return gen;
}
