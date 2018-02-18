#include "bnmf_algs.hpp"

#include <limits>


std::pair<Eigen::MatrixXd, Eigen::MatrixXd> bnmf_algs::nmf_euclidean(const Eigen::MatrixXd& X, long r, int max_iter) {
    using namespace Eigen;
    constexpr int num_prev_costs = 5;
    const long m = X.rows();
    const long n = X.cols();

    // initialize
    MatrixXd W = MatrixXd::Random(m, r) + MatrixXd::Ones(m, r);
    MatrixXd H = MatrixXd::Random(r, n) + MatrixXd::Ones(r, n);

    double cost_arr[num_prev_costs];
    double cost_sum = 0., prev_sum, prev_cost;
    int cost_index = 0;
    while (max_iter-- > 0) {
        MatrixXd curr_approx = W*H;

        // update costs
        prev_cost = cost_arr[cost_index];
        prev_sum = cost_sum;
        cost_arr[cost_index] = (X - curr_approx).norm();
        cost_index = (cost_index + 1)%num_prev_costs;
        cost_sum = cost_sum + cost_arr[cost_index] - prev_cost;

        // check cost convergence
        if (std::abs(cost_sum - prev_sum) < std::numeric_limits<double>::epsilon()) {
            break;
        }
        // H update
        MatrixXd numer = W.transpose()*X;
        MatrixXd denom = W.transpose()*curr_approx;
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < n; ++j) {
                H(i, j) *= numer(i, j)/denom(i, j);
            }
        }

        // W update
        numer = X*H.transpose();
        denom = W*H*H.transpose();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < r; ++j) {
                W(i, j) *= numer(i, j)/denom(i, j);
            }
        }
    }
    return {W, H};
}
