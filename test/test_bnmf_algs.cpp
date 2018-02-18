#include "catch2.hpp"
#include "bnmf_algs.hpp"
#include <fstream>
#include <iostream>

TEST_CASE("Example test case", "[example]") {
    using namespace Eigen;

    MatrixXd X = MatrixXd::Random(500, 400) + MatrixXd::Ones(500, 400);
    MatrixXd W, H;
    std::tie(W, H) = bnmf_algs::nmf_euclidean(X, 400);
    std::ofstream ofs("output.txt", std::ios_base::trunc);
    ofs << "Matrix X\n" << X << '\n';
    ofs << "Matrix W\n" << W << '\n';
    ofs << "Matrix H\n" << H << std::endl;
    ofs << "Matrix WH\n" << W*H << std::endl;
    std::cout << "Test completed" << std::endl;
    std::cout << "Euclidean norm: " << (X - W*H).norm() << std::endl;
}
