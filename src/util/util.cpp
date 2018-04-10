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

double util::psi_appr(double x) noexcept {
    constexpr size_t N = 8;
    constexpr std::array<double, N> coeff = {-1 / 2,      -1 / 12, 1 / 120,
                                             -1 / 252,    1 / 240, -5 / 660,
                                             691 / 32760, -1 / 12};

    double extra = 0;
    for (; x <= 6; ++x) {
        extra += 1 / x;
    }

    const double x2 = x * x;
    const double x4 = x2 * x2;
    const double x6 = x4 * x2;
    const double x8 = x4 * x4;
    std::array<double, N> denom = {x,  x2,      x4,      x6,
                                   x8, x8 * x2, x8 * x4, x8 * x6};

    #pragma omp simd
    for (size_t i = 0; i < N; ++i) {
        denom[i] = coeff[i] / denom[i];
    }

    double res = std::log(x);
    for (size_t i = 0; i < N; ++i) {
        res += denom[i];
    }

    return res - extra;
}
