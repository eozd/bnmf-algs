#include "../catch2.hpp"
#include "util/util.hpp"
#include <iostream>

using namespace bnmf_algs;

int min(int a, int b) { return a < b ? a : b; }

TEST_CASE("Call using std::tuple", "[call]") {
    auto tuple = std::make_tuple(5, 20);
    REQUIRE(util::call(min, tuple) == 5);
}

TEST_CASE("Call using std::array", "[call]") {
    std::array<int, 2> arr{5, 10};
    REQUIRE(util::call(min, arr) == 5);
}

TEST_CASE("Call using std::pair", "[call]") {
    std::pair<int, int> p{5, 10};
    REQUIRE(util::call(min, p) == 5);
}

TEST_CASE("Call using selected elements", "[call]") {
    // Use 0th and 2nd indices
    std::index_sequence<0, 2> indices;
    auto tuple = std::make_tuple(1, "aoesu", 2);
    REQUIRE(util::call(min, tuple, indices) == 1);
}

TEST_CASE("Test sparseness", "[sparseness]") {
    size_t x = 5, y = 5, z = 5;
    shape<3> tensor_shape{x, y, z};

    SECTION("Zero tensor") {
        tensord<3> S(x, y, z);
        S.setZero();
        REQUIRE(util::sparseness(S) == std::numeric_limits<double>::max());
    }

    SECTION("Tensor with a single nonzero element") {
        tensord<3> S(x, y, z);
        S.setZero();
        S(0, 0, 0) = 240;
        REQUIRE(util::sparseness(S) == Approx(1));
    }

    SECTION("Tensor with all ones") {
        tensord<3> S(x, y, z);
        S.setConstant(1);
        double result = util::sparseness(S);
        REQUIRE(Approx(result).margin(std::numeric_limits<double>::epsilon()) ==
                0);
    }
}

TEST_CASE("Test normalize", "[normalize] [normalized]") {
    int x = 2, y = 10, z = 3;
    tensord<3> S(x, y, z);
    S.setValues({{{-3., 0., 0.},
                  {1., 0., 0.},
                  {-1., 0., 1.},
                  {1., 0., -3.},
                  {0., 0., 0.},
                  {-3., 0., 0.},
                  {0., 0., -2.},
                  {1., 0., 0.},
                  {2., 0., 0.},
                  {0., 0., 2.}},

                 {{2., 0., 0.},
                  {5., 0., 0.},
                  {9., 0., 0.},
                  {5., 0., -1.},
                  {6., 0., 0.},
                  {7., 0., 0.},
                  {9., -2., 0.},
                  {6., 0., 0.},
                  {4., 0., 0.},
                  {2., 0., 0.}}});

    SECTION("Test invalid axis") {
        size_t axis = 3;
        REQUIRE_THROWS(util::normalize(S, axis));
        REQUIRE_THROWS(util::normalized(S, axis));
        axis = 2;
        REQUIRE_NOTHROW(util::normalize(S, axis));
        REQUIRE_NOTHROW(util::normalized(S, axis));
        axis = 0;
        REQUIRE_NOTHROW(util::normalize(S, axis));
        REQUIRE_NOTHROW(util::normalized(S, axis));
    }
    SECTION("Test tensor dimensions") {
        // does it fail to compile as expected?
        // tensord<0> zero_tensor;
        // util::normalize(zero_tensor, 0);

        tensord<1> tensord1(1);
        tensord1.setValues({5});
        util::normalize(tensord1, 0);
        REQUIRE(tensord1(0) == 1);
    }
    SECTION("Test L1 norm results for axis 0") {
        tensord<3> res(S.dimensions());
        res.setValues({{{-0.6, 0., 0.},
                        {0.16666667, 0., 0.},
                        {-0.1, 0., 1.},
                        {0.16666667, 0., -0.75},
                        {0., 0., 0.},
                        {-0.3, 0., 0.},
                        {0., 0., -1.},
                        {0.14285714, 0., 0.},
                        {0.33333333, 0., 0.},
                        {0., 0., 1.}},

                       {{0.4, 0., 0.},
                        {0.83333333, 0., 0.},
                        {0.9, 0., 0.},
                        {0.83333333, 0., -0.25},
                        {1., 0., 0.},
                        {0.7, 0., 0.},
                        {1., -1., 0.},
                        {0.85714286, 0., 0.},
                        {0.66666667, 0., 0.},
                        {1., 0., 0.}}});
        util::normalize(S, 0);
        Eigen::Map<vector_t> vec_r(res.data(), res.size());
        Eigen::Map<vector_t> vec_s(S.data(), S.size());
        REQUIRE(vec_r.isApprox(vec_s, 1e-8));
    }

    SECTION("Test L1 norm results for axis 1") {
        tensord<3> res(S.dimensions());
        res.setValues({{{-0.25, 0., 0.},
                        {0.08333333, 0., 0.},
                        {-0.08333333, 0., 0.125},
                        {0.08333333, 0., -0.375},
                        {0., 0., 0.},
                        {-0.25, 0., 0.},
                        {0., 0., -0.25},
                        {0.08333333, 0., 0.},
                        {0.16666667, 0., 0.},
                        {0., 0., 0.25}},

                       {{0.03636364, 0., 0.},
                        {0.09090909, 0., 0.},
                        {0.16363636, 0., 0.},
                        {0.09090909, 0., -1.},
                        {0.10909091, 0., 0.},
                        {0.12727273, 0., 0.},
                        {0.16363636, -1., 0.},
                        {0.10909091, 0., 0.},
                        {0.07272727, 0., 0.},
                        {0.03636364, 0., 0.}}});
        util::normalize(S, 1);
        Eigen::Map<vector_t> vec_r(res.data(), res.size());
        Eigen::Map<vector_t> vec_s(S.data(), S.size());
        REQUIRE(vec_r.isApprox(vec_s, 1e-8));
    }

    SECTION("Test L1 norm results for axis 2") {
        tensord<3> res(S.dimensions());
        res.setValues({{{-1., 0., 0.},
                        {1., 0., 0.},
                        {-0.5, 0., 0.5},
                        {0.25, 0., -0.75},
                        {0., 0., 0.},
                        {-1., 0., 0.},
                        {0., 0., -1.},
                        {1., 0., 0.},
                        {1., 0., 0.},
                        {0., 0., 1.}},

                       {{1., 0., 0.},
                        {1., 0., 0.},
                        {1., 0., 0.},
                        {0.83333333, 0., -0.16666667},
                        {1., 0., 0.},
                        {1., 0., 0.},
                        {0.81818182, -0.18181818, 0.},
                        {1., 0., 0.},
                        {1., 0., 0.},
                        {1., 0., 0.}}});
        util::normalize(S, 2);
        Eigen::Map<vector_t> vec_r(res.data(), res.size());
        Eigen::Map<vector_t> vec_s(S.data(), S.size());
        REQUIRE(vec_r.isApprox(vec_s, 1e-8));
    }

    S.setValues({{{-5., 1., 0.},
                  {6., 0., 9.},
                  {0., 2., 1.},
                  {3., 1., 0.},
                  {2., 2., 1.},
                  {1., 2., 3.},
                  {0., 1., 0.},
                  {6., 0., 16.},
                  {0., 28., 0.},
                  {19., 0., 15.}},

                 {{0., 10., -4.},
                  {26., 0., 5.},
                  {0., 17., 0.},
                  {3., 5., 1.},
                  {1., 0., 0.},
                  {2., -204., 2.},
                  {4., 0., 5.},
                  {0., 1., 1.},
                  {3., 1., 4.},
                  {1., 4., 0.}}});

    SECTION("Test L2 norm results for axis 0") {
        tensord<3> res(S.dimensions());
        res.setValues({{{-0.2, 0.00990099, 0.},
                        {0.00842697, 0., 0.08490566},
                        {0., 0.00682594, 1.},
                        {0.16666667, 0.03846154, 0.},
                        {0.4, 0.5, 1.},
                        {0.2, 0.00004805, 0.23076923},
                        {0., 1., 0.},
                        {0.16666667, 0., 0.06225681},
                        {0., 0.03566879, 0.},
                        {0.05248619, 0., 0.06666667}},

                       {{0., 0.0990099, -0.25},
                        {0.03651685, 0., 0.04716981},
                        {0., 0.05802048, 0.},
                        {0.16666667, 0.19230769, 1.},
                        {0.2, 0., 0.},
                        {0.4, -0.00490149, 0.15384615},
                        {0.25, 0., 0.2},
                        {0., 1., 0.00389105},
                        {0.33333333, 0.00127389, 0.25},
                        {0.00276243, 0.25, 0.}}});
        util::normalize(S, 0, util::NormType::L2);
        Eigen::Map<vector_t> vec_r(res.data(), res.size());
        Eigen::Map<vector_t> vec_s(S.data(), S.size());
        REQUIRE(vec_r.isApprox(vec_s, 1e-8));
    }
    SECTION("Test L2 norm results for axis 1") {
        tensord<3> res(S.dimensions());
        res.setValues({{{-0.0105932203, 0.0012515645, 0.},
                        {0.0127118644, 0., 0.0157068063},
                        {0., 0.0025031289, 0.0017452007},
                        {0.0063559322, 0.0012515645, 0.},
                        {0.0042372881, 0.0025031289, 0.0017452007},
                        {0.0021186441, 0.0025031289, 0.0052356021},
                        {0., 0.0012515645, 0.},
                        {0.0127118644, 0., 0.0279232112},
                        {0., 0.0350438048, 0.},
                        {0.0402542373, 0., 0.0261780105}},

                       {{0., 0.0002378234, -0.0454545455},
                        {0.0363128492, 0., 0.0568181818},
                        {0., 0.0004042998, 0.},
                        {0.0041899441, 0.0001189117, 0.0113636364},
                        {0.001396648, 0., 0.},
                        {0.0027932961, -0.0048515982, 0.0227272727},
                        {0.0055865922, 0., 0.0568181818},
                        {0., 0.0000237823, 0.0113636364},
                        {0.0041899441, 0.0000237823, 0.0454545455},
                        {0.001396648, 0.0000951294, 0.}}});
        util::normalize(S, 1, util::NormType::L2);
        Eigen::Map<vector_t> vec_r(res.data(), res.size());
        Eigen::Map<vector_t> vec_s(S.data(), S.size());

        REQUIRE(vec_r.isApprox(vec_s, 1e-8));
    }
    SECTION("Test Max norm results for axis 2") {
        tensord<3> res(S.dimensions());
        res.setValues({{{-1., 0.2, 0.},
                        {0.6666666667, 0., 1.},
                        {0., 1., 0.5},
                        {1., 0.3333333333, 0.},
                        {1., 1., 0.5},
                        {0.3333333333, 0.6666666667, 1.},
                        {0., 1., 0.},
                        {0.375, 0., 1.},
                        {0., 1., 0.},
                        {1., 0., 0.7894736842}},

                       {{0., 1., -0.4},
                        {1., 0., 0.1923076923},
                        {0., 1., 0.},
                        {0.6, 1., 0.2},
                        {1., 0., 0.},
                        {0.0098039216, -1., 0.0098039216},
                        {0.8, 0., 1.},
                        {0., 1., 1.},
                        {0.75, 0.25, 1.},
                        {0.25, 1., 0.}}});
        util::normalize(S, 2, util::NormType::Max);
        Eigen::Map<vector_t> vec_r(res.data(), res.size());
        Eigen::Map<vector_t> vec_s(S.data(), S.size());

        REQUIRE(vec_r.isApprox(vec_s, 1e-8));
    }

    SECTION("Test normalized") {
        auto V = util::normalized(S, 1, util::NormType::L2);
        util::normalize(S, 1, util::NormType::L2);

        // require exact equality
        tensorx<bool, 0> res = (V == S).all();
        REQUIRE(res.coeff());
    }
}
