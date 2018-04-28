#include "bnmf_algs.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std::chrono;
using namespace bnmf_algs;

/**
 * @brief Read a matrix X from a given file, run a factorization algorithm on it,
 * and write the resulting two matrices to W.txt and H.txt, and the resulting
 * tensor to S.txt files. If NMF is run, only W.txt and H.txt contain meaningful
 * results.
 *
 * Given input file must contain each row of the matrix on a separated line.
 * Each entry in a row must be separated using a single space character.
 */
int main(int argc, char** argv) {
    // TODO: Improve file parsing code to allow scientific notation and multiple
    if (argc != 6) {
        std::cout << "usage: " << argv[0]
                  << " alg filename n_components beta max_iter\n\n"
                  << "where alg is one of <nmf | seq_greedy_bld | bld_mult | "
                     "bld_mult_cuda | bld_add | bld_appr | collapsed_gibbs | "
                     "collapsed_icm>"
                  << std::endl;
        return -1;
    }
    std::string alg(argv[1]);
    std::string filename(argv[2]);
    size_t n_components = std::stoul(argv[3]);
    double beta = std::stod(argv[4]);
    size_t max_iter = std::stoul(argv[5]);

    std::ifstream datafile(filename);
    std::vector<double> data;
    int n_rows = 0;
    std::string line;
    double value;
    while (std::getline(datafile, line)) {
        size_t index = 0, prev_index = 0;
        index = line.find(' ');
        while (prev_index != std::string::npos) {
            value =
                std::atof(line.substr(prev_index, index - prev_index).data());
            data.push_back(value);

            prev_index = index;
            index = line.find(' ', index + 1);
        }
        ++n_rows;
    }
    auto n_cols = data.size() / n_rows;

    matrixd X = Eigen::Map<matrixd>(data.data(), n_rows, n_cols);

    std::vector<double> alpha_dirichlet(X.rows(), 0.05);
    std::vector<double> beta_dirichlet(n_components, 10);
    beta_dirichlet.back() = 60;
    alloc_model::Params<double> params(100, 1, alpha_dirichlet,
                                              beta_dirichlet);
    // std::vector<double> alpha_dirichlet(X.rows(), 1);
    // std::vector<double> beta_dirichlet(n_components, 1);
    // alloc_model::Params<double> params(40, 1, alpha_dirichlet,
    //                                          beta_dirichlet);

    std::cout << "Computing the factorization" << std::endl;
    auto alg_begin_time = high_resolution_clock::now();
    tensord<3> S;
    matrixd W, H;
    vectord L;
    if (alg == "nmf") {
        std::tie(W, H) = nmf::nmf(X, n_components, beta, max_iter);
    } else if (alg == "seq_greedy_bld") {
        S = bld::seq_greedy_bld(X, n_components, params);
        std::tie(W, H, L) = bld::bld_fact(S, params);
    } else if (alg == "bld_mult") {
        S = bld::bld_mult(X, n_components, params, max_iter, true);
        std::tie(W, H, L) = bld::bld_fact(S, params);
#ifdef USE_CUDA
    } else if (alg == "bld_mult_cuda") {
        S = bld::bld_mult_cuda(X, n_components, params, max_iter, true);
        std::tie(W, H, L) = bld::bld_fact(S, params);
#endif
    } else if (alg == "bld_add") {
        S = bld::bld_add(X, n_components, params, max_iter);
        std::tie(W, H, L) = bld::bld_fact(S, params);
    } else if (alg == "bld_appr") {
        S = std::get<0>(bld::bld_appr(X, n_components, params, max_iter));
        std::tie(W, H, L) = bld::bld_fact(S, params);
    } else if (alg == "collapsed_gibbs") {
        auto gen = bld::collapsed_gibbs(X, n_components, params, max_iter);
        for (const auto& _ : gen)
            ;
        S = *gen.begin();
        std::tie(W, H, L) = bld::bld_fact(S, params);
    } else if (alg == "collapsed_icm") {
        S = bld::collapsed_icm(X, n_components, params, max_iter);
        std::tie(W, H, L) = bld::bld_fact(S, params);
    } else {
        std::cout << "Invalid algorithm" << std::endl;
        return -1;
    }
    auto alg_end_time = high_resolution_clock::now();
    std::cout
        << "Total time: "
        << duration_cast<milliseconds>(alg_end_time - alg_begin_time).count()
        << " milliseconds" << std::endl;

    std::ofstream s_file("S.txt", std::ios_base::trunc);
    std::ofstream w_file("W.txt", std::ios_base::trunc);
    std::ofstream h_file("H.txt", std::ios_base::trunc);
    std::ofstream l_file("L.txt", std::ios_base::trunc);
    s_file << S;
    w_file << W;
    h_file << H;
    l_file << L;
}
