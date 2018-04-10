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
    double extra = 0;

    // write each case separately to minimize number of divisions as much as
    // possible
    if (x < 1) {
        const double a = x + 1;
        const double b = x + 2;
        const double c = x + 3;
        const double d = x + 4;
        const double e = x + 5;
        const double ab = a * b;
        const double cd = c * d;
        const double ex = e * x;

        extra = ((a + b) * cd * ex + ab * d * ex + ab * c * ex + ab * cd * x +
                 ab * cd * e) /
                (ab * cd * ex);
        x += 6;
    } else if (x < 2) {
        const double a = x + 1;
        const double b = x + 2;
        const double c = x + 3;
        const double d = x + 4;
        const double ab = a * b;
        const double cd = c * d;
        const double dx = d * x;

        extra =
            ((a + b) * c * dx + ab * dx + ab * c * x + ab * cd) / (ab * cd * x);
        x += 5;
    } else if (x < 3) {
        const double a = x + 1;
        const double b = x + 2;
        const double c = x + 3;
        const double ab = a * b;
        const double cx = c * x;

        extra = ((a + b) * cx + (c + x) * ab) / (ab * cx);
        x += 4;
    } else if (x < 4) {
        const double a = x + 1;
        const double b = x + 2;
        const double ab = a * b;

        extra = ((a + b) * x + ab) / (ab * x);
        x += 3;
    } else if (x < 5) {
        const double a = x + 1;

        extra = (a + x) / (a * x);
        x += 2;
    } else if (x < 6) {
        extra = 1 / x;
        x += 1;
    }

    double x2 = x * x;
    double x4 = x2 * x2;
    double x6 = x4 * x2;
    double x8 = x6 * x2;
    double x10 = x8 * x2;
    double x12 = x10 * x2;
    double x13 = x12 * x;
    double x14 = x13 * x;

    // write the result of the formula simplified using symbolic calculation
    // to minimize the number of divisions
    double res =
        std::log(x) + (-360360 * x13 - 60060 * x12 + 6006 * x10 - 2860 * x8 +
                       3003 * x6 - 5460 * x4 + 15202 * x2 - 60060) /
                          (720720 * x14);

    return res - extra;
}
