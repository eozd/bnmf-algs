#include <library.hpp>

#include <iostream>
#include <Eigen/Dense>
#include <gsl/gsl_math.h>


int bnmf_algs::hello(int x) {
    std::cout << "Hello, World!" << x << std::endl;
    Eigen::MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;
    std::cout << gsl_pow_int(3.7, 4) << std::endl;
    return x;
}