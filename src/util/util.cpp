#include "util/util.hpp"
#include "util/generator.hpp"
#include "util/wrappers.hpp"


using namespace bnmf_algs;

double util::sparseness(const tensord<3>& S) {
    // TODO: implement a method to unpack std::array
    long x = S.dimension(0), y = S.dimension(1), z = S.dimension(2);
    double sum = 0, squared_sum = 0;

    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            for (int k = 0; k < z; ++k) {
                sum += S(i, j, k);
                squared_sum += S(i, j, k) * S(i, j, k);
            }
        }
    }
    if (squared_sum < std::numeric_limits<double>::epsilon()) {
        return std::numeric_limits<double>::max();
    }
    double frob_norm = std::sqrt(squared_sum);
    double axis_mult = std::sqrt(x * y * z);

    return (axis_mult - sum / frob_norm) / (axis_mult - 1);
}

details::SampleOnesComputer::SampleOnesComputer(const matrix_t& X,
                                                bool replacement)
    : replacement(replacement), X_cumsum(vector_t(X.cols() * X.rows())),
      X_cols(X.cols()),
      X_sum(replacement ? X.array().sum() : std::floor(X.array().sum())),
      rnd_gen(util::make_gsl_rng(gsl_rng_taus)) {

    auto X_arr = X.array();
    X_cumsum(0) = X_arr(0);
    for (int i = 1; i < X_arr.size(); ++i) {
        X_cumsum(i) = X_cumsum(i - 1) + X_arr(i);
    }
}

void details::SampleOnesComputer::operator()(size_t curr_step,
                                             std::pair<int, int>& prev_val) {
    double u = gsl_ran_flat(rnd_gen.get(), 0, 1);

    vector_t cum_prob;
    if (replacement) {
        cum_prob = X_cumsum.array() / X_sum;
    } else {
        cum_prob = X_cumsum.array() / (X_sum - curr_step);
    }

    double* begin = cum_prob.data();
    double* end = cum_prob.data() + cum_prob.cols();
    auto it = std::upper_bound(begin, end, u);

    long m = it - begin;

    if (!replacement) {
        vector_t diff = vector_t::Constant(X_cumsum.cols(), 1);
        diff.head(m) = vector_t::Zero(m);
        X_cumsum -= diff;
    }

    prev_val.first = static_cast<int>(m / X_cols);
    prev_val.second = static_cast<int>(m % X_cols);
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
