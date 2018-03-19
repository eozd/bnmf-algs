#include "defs.hpp"
#include "nmf/nmf.hpp"
#include <fstream>
#include <iostream>
#include <vector>

using namespace bnmf_algs;

/**
 * @brief Read a matrix X from a given file, run NMF on it, and write the
 * resulting two matrices to W.txt and H.txt files.
 *
 * Given input file must contain each row of the matrix on a separated line.
 * Each entry in a row must be separated using a single space character.
 */
int main(int argc, char** argv) {
    // TODO: Improve file parsing code to allow scientific notation and multiple
    if (argc != 5) {
        std::cout << "usage: " << argv[0]
                  << " filename n_components beta max_iter" << std::endl;
        return -1;
    }
    std::string filename(argv[1]);
    size_t n_components = std::stoul(argv[2]);
    double beta = std::stod(argv[3]);
    size_t max_iter = std::stoul(argv[4]);

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

    Eigen::Map<matrix_t> X(data.data(), n_rows, n_cols);

    matrix_t W, H;
    std::tie(W, H) = nmf::nmf(X, n_components, beta, max_iter);

    std::ofstream w_file("W.txt", std::ios_base::trunc);
    std::ofstream h_file("H.txt", std::ios_base::trunc);
    w_file << W;
    h_file << H;
}
