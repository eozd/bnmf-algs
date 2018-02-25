
#include "decomposition.hpp"
#include "catch2.hpp"
#include "sampling.hpp"
#include "util.hpp"
#include <iostream>

using namespace bnmf_algs;
using namespace bnmf_algs::util;

TEST_CASE("Parameter checks on seq_greedy_bld", "[seq_greedy_bld]") {
	int x = 10, y = 8, z = 3;
    matrix_t X = matrix_t::Random(x, y) + matrix_t::Constant(x, y, 5);
	std::vector<double> alpha(x, 1);
	std::vector<double> beta(z, 1);

	REQUIRE_NOTHROW(seq_greedy_bld(X, z, alpha, beta));

	alpha.resize(x - 1);
	REQUIRE_THROWS(seq_greedy_bld(X, z, alpha, beta));

    alpha.resize(x);
	beta.resize(z - 1);
	REQUIRE_THROWS(seq_greedy_bld(X, z, alpha, beta));

	alpha.resize(x + 1);
	REQUIRE_THROWS(seq_greedy_bld(X, z, alpha, beta));
}

