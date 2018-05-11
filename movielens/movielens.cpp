#include "bnmf_algs.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>

using namespace std::chrono;
using namespace bnmf_algs;

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " filename n_components max_iter"
                  << std::endl;
        return -1;
    }
    const std::string filename(argv[1]);
    const size_t n_components = std::stoul(argv[2]);
    const size_t max_iter = std::stoul(argv[3]);

    std::ifstream datafile(filename);
    assert(datafile);
    constexpr size_t n_rows = 943, n_cols = 1682;

    matrix_t<double> X = matrix_t<double>::Constant(n_rows, n_cols, NAN);
    int value;
    size_t user_id, movie_id, timestamp;
    while (datafile >> user_id) {
        datafile >> movie_id >> value >> timestamp;
        X(user_id - 1, movie_id - 1) = value;
    }

    {
        std::ofstream x_orig_file("X_orig.txt", std::ios_base::trunc);
        x_orig_file << X;
    }

	std::vector<double> alpha(n_rows, 0.01);
	std::vector<double> beta(n_cols, 1000);
	double b = 2;
	alloc_model::Params<double> params(1, b, alpha, beta);
	std::vector<alloc_model::Params<double>> param_vec(n_components, params);

    std::cout << "Computing the factorization" << std::endl;
    auto alg_begin_time = high_resolution_clock::now();

    // without psi approximate
    auto em_res = bld::online_EM(X, param_vec, max_iter, false);

    auto alg_end_time = high_resolution_clock::now();
    std::cout
        << "Total time: "
        << duration_cast<milliseconds>(alg_end_time - alg_begin_time).count()
        << " milliseconds" << std::endl;

	for (size_t i = 0; i < em_res.EM_bound.cols(); ++i) {
		std::cout << em_res.EM_bound(i) << std::endl;
	}

	matrix_t<double> W = em_res.logW.array().exp();
    matrix_t<double> H = em_res.logH.array().exp();
    tensor_t<double, 3> S(n_rows, n_cols, n_components);
    for (size_t i = 0; i < n_rows; ++i) {
        for (size_t j = 0; j < n_cols; ++j) {
            for (size_t k = 0; k < n_components; ++k) {
                S(i, j, k) = W(i, k) * H(k, j);
            }
        }
    }

    std::ofstream x_file("X_full.txt", std::ios_base::trunc);
    std::ofstream h_file("W.txt", std::ios_base::trunc);
    std::ofstream w_file("H.txt", std::ios_base::trunc);
    std::ofstream s_file("S.txt", std::ios_base::trunc);

    x_file << em_res.X_full;
    w_file << W;
    h_file << H;
    s_file << S;
}
