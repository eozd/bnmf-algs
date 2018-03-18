#include "defs.hpp"
#include "nmf/nmf.hpp"
#include <fstream>
#include <iostream>
#include <vector>

/**
 * @brief Read a matrix X from a given file, run NMF on it, and write the
 * resulting two matrices to W.txt and H.txt files.
 *
 * Given input file must contain each row of the matrix on a separated line.
 * Each entry in a row must be separated using a single space character.
 */
int main() {
    // TODO: Get the file name from console arguments
    // TODO: Improve file parsing code to allow scientific notation and multiple
    // spaces
    std::ifstream datafile("olivetti.txt");
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
    std::cout << '(' << n_rows << ", " << n_cols << ')' << std::endl;

    Eigen::Map<bnmf_algs::matrix_t> X(data.data(), n_rows, n_cols);

    bnmf_algs::matrix_t W, H;
    std::tie(W, H) =
        bnmf_algs::nmf::nmf(X, 64, 1, 2000);

    std::ofstream w_file("W.txt", std::ios_base::trunc);
    std::ofstream h_file("H.txt", std::ios_base::trunc);
    w_file << W;
    h_file << H;
}
