#include "util/sampling.hpp"

using namespace bnmf_algs;

details::SampleOnesNoReplaceComputer::SampleOnesNoReplaceComputer(
    const matrix_t& X)
    : vals_indices(),
      no_repl_comp([](const val_index& left, const val_index& right) {
          return left.first < right.first;
      }) {

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

void details::SampleOnesNoReplaceComputer::
operator()(size_t curr_step, std::pair<int, int>& prev_val) {
    // max heap sampling (deterministic)
    if (vals_indices.empty()) {
        return;
    }
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

details::SampleOnesReplaceComputer::SampleOnesReplaceComputer(const matrix_t& X)
    : cum_prob(), X_cols(X.cols()), X_sum(X.array().sum()),
      rnd_gen(util::make_gsl_rng(gsl_rng_taus)) {

    cum_prob = vector_t(X.rows() * X.cols());
    auto X_arr = X.array();
    cum_prob(0) = X_arr(0);
    for (int i = 1; i < X_arr.size(); ++i) {
        cum_prob(i) = cum_prob(i - 1) + X_arr(i);
    }
}

void details::SampleOnesReplaceComputer::
operator()(size_t curr_step, std::pair<int, int>& prev_val) {
    // inverse CDF sampling
    double* begin = cum_prob.data();
    double* end = cum_prob.data() + cum_prob.cols();
    auto it = util::choice(begin, end, rnd_gen);
    long m = it - begin;

    prev_val.first = static_cast<int>(m / X_cols);
    prev_val.second = static_cast<int>(m % X_cols);
}

static std::string check_sample_ones_params(const matrix_t& X) {
    if (X.array().size() == 0) {
        return "Matrix X must have at least one element";
    }
    if ((X.array() < 0).any()) {
        return "Matrix X must be nonnegative";
    }
    if ((X.array().abs() <= std::numeric_limits<double>::epsilon()).all()) {
        return "Matrix X must have at least one nonzero element";
    }
    return "";
}

util::Generator<std::pair<int, int>, details::SampleOnesNoReplaceComputer>
util::sample_ones_noreplace(const matrix_t& X) {
    {
        auto error_msg = check_sample_ones_params(X);
        if (!error_msg.empty()) {
            throw std::invalid_argument(error_msg);
        }
    }

    size_t num_samples = static_cast<size_t>(X.array().sum());
    // don't know the initial value
    std::pair<int, int> init_val;
    util::Generator<std::pair<int, int>, details::SampleOnesNoReplaceComputer>
        gen(init_val, num_samples + 1,
            details::SampleOnesNoReplaceComputer(X));

    // get rid of the first empty value
    ++gen.begin();

    return gen;
}

util::Generator<std::pair<int, int>, details::SampleOnesReplaceComputer>
util::sample_ones_replace(const matrix_t& X, size_t num_samples) {
    {
        auto error_msg = check_sample_ones_params(X);
        if (!error_msg.empty()) {
            throw std::invalid_argument(error_msg);
        }
    }

    // don't know the initial value
    std::pair<int, int> init_val;
    util::Generator<std::pair<int, int>, details::SampleOnesReplaceComputer>
            gen(init_val, num_samples + 1,
                details::SampleOnesReplaceComputer(X));

    // get rid of the first empty value
    ++gen.begin();

    return gen;
}
