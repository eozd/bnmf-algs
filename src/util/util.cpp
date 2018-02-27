#include "util/util.hpp"
#include "defs.hpp"

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
