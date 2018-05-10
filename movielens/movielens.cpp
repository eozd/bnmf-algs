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

    shape<3> tensor_shape{n_rows, n_cols, n_components};
    auto param_vec = alloc_model::make_EM_params<double>(tensor_shape);

    std::cout << "Computing the factorization" << std::endl;
    auto alg_begin_time = high_resolution_clock::now();

    // without psi approximate
    auto em_res = bld::online_EM(X, param_vec, max_iter, false);

    auto alg_end_time = high_resolution_clock::now();
    std::cout
        << "Total time: "
        << duration_cast<milliseconds>(alg_end_time - alg_begin_time).count()
        << " milliseconds" << std::endl;

    std::ofstream x_file("X_full.txt", std::ios_base::trunc);
    x_file << em_res.X_full;
}
