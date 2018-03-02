#include "../catch2.hpp"
#include "defs.hpp"
#include "util/generator.hpp"
#include <algorithm>
#include <iostream>
#include <memory>

using namespace bnmf_algs::util;

void str_computer(size_t index, std::string& prev) { prev += "abc"; }

struct C {
    static size_t TotalDefaultCtorCount;
    static size_t TotalCopyCount;

    static void reset_counters() {
        TotalDefaultCtorCount = 0;
        TotalCopyCount = 0;
    }

    int val;
    C() : val(0) { ++TotalDefaultCtorCount; }

    C(const C& other) {
        this->val = other.val;
        ++TotalCopyCount;
    }

    C& operator=(const C& other) {
        this->val = other.val;
        ++TotalCopyCount;
        return *this;
    }
};

size_t C::TotalDefaultCtorCount = 0;
size_t C::TotalCopyCount = 0;

void c_computer(size_t index, C& prev) { prev.val++; }

struct Op {
    void operator()(size_t index, C& prev) { prev.val++; }
};

TEST_CASE("Test generator range-based for loop", "[generator]") {
    std::vector<int> nums;
    for (const auto& c_obj : Generator<C, Op>(C(), 500, Op())) {
        nums.push_back(c_obj.val);
    }

    std::vector<int> expected(nums.size());
    std::iota(expected.begin(), expected.end(), 0);

    REQUIRE(nums == expected);
}

TEST_CASE("Test generator begin/end iterator loop", "[generator]") {
    std::vector<int> nums;
    Generator<C, decltype(&c_computer)> gen(C(), 500, c_computer);
    for (auto it = gen.begin(); it != gen.end(); ++it) {
        nums.push_back(it->val);
    }

    std::vector<int> expected(nums.size());
    std::iota(expected.begin(), expected.end(), 0);

    REQUIRE(nums == expected);
}

TEST_CASE("Test generator consuming", "[generator]") {
    SECTION("Test if no values are generated using range-based for loop") {
        Generator<C, Op> gen(C(), 500, Op());
        // consume the generator
        for (const auto& c_obj : gen)
            ;

        std::vector<int> nums;
        for (const auto& c_obj : gen) {
            nums.push_back(c_obj.val);
        }

        std::vector<int> expected(0);
        REQUIRE(nums == expected);
    }

    SECTION("Test if no values are generated using iterator based for loop") {
        Generator<C, Op> gen(C(), 500, Op());
        // consume the generator
        for (auto it = gen.begin(); it != gen.end(); ++it)
            ;

        std::vector<int> nums;
        for (auto it = gen.begin(); it != gen.end(); ++it) {
            nums.push_back(it->val);
        }

        std::vector<int> expected(0);
        REQUIRE(nums == expected);
    }

    SECTION("Test if values can be generated manually after gen is consumed") {
        Generator<C, Op> gen(C(), 500, Op());
        // consume the generator
        for (auto it = gen.begin(); it != gen.end(); ++it)
            ;

        std::vector<int> nums;
        int count = 0;
        // generate 10 more values from where the computation was left off
        for (auto it = gen.begin(); count < 10; ++it, ++count) {
            nums.push_back(it->val);
        }

        std::vector<int> expected(10);
        std::iota(expected.begin(), expected.end(), 500);

        REQUIRE(nums == expected);
    }
}

TEST_CASE("Test ComputationIterator constructs or copies no additional objects",
          "[ComputationIterator]") {
    C c_obj;
    C::reset_counters();

    size_t beg = 0, end_count = 500;
    Op oper;
    ComputationIterator<C, Op> begin(&c_obj, &beg, &oper);
    ComputationIterator<C, Op> end(&end_count);

    for (auto it = begin; it != end; ++it) {
        it->val;
    }
    REQUIRE(C::TotalDefaultCtorCount == 0);
    REQUIRE(C::TotalCopyCount == 0);
}

TEST_CASE("Test Generator makes only a single copy for itself", "[generator]") {
    C c_obj;
    C::reset_counters();

    Generator<C, Op> gen(c_obj, 1000, Op());
    for (const auto& c_obj : gen)
        ;
    REQUIRE(C::TotalDefaultCtorCount == 0);
    REQUIRE(C::TotalCopyCount == 1);
}

TEST_CASE("Test if incrementing a copy of iterator updates original value",
          "[ComputationIterator]") {
    C c_obj;

    size_t beg = 0, end_count = 500;
    Op increment;
    ComputationIterator<C, Op> begin(&c_obj, &beg, &increment);

    auto it = begin;

    REQUIRE(it->val == 0);
    REQUIRE(begin->val == 0);

    ++it;

    REQUIRE(it->val == 1);
    REQUIRE(begin->val == 1);
}

void increment(size_t index, int& prev) { ++prev; }

TEST_CASE("Test various STL algorithms with Generator", "[generator]") {
    SECTION("std::find") {
        Generator<int, decltype(&increment)> gen(0, 500, increment);

        auto it = std::find(gen.begin(), gen.end(), 333);
        REQUIRE(*it == 333);

        // generator has consumed all the values before 333; it cannot find them
        // anymore
        it = std::find(gen.begin(), gen.end(), 120);
        REQUIRE(*it != 120);

        // since nothing was found, generator was consumed to the end. Now,
        // nothing can be found
        it = std::find(gen.begin(), gen.end(), 482);
        REQUIRE(*it != 482);
    }

    SECTION("std::copy") {
        size_t count = 500;
        Generator<int, decltype(&increment)> gen(0, count, increment);

        std::vector<int> computation_trace(count);
        std::copy(gen.begin(), gen.end(), computation_trace.begin());

        std::vector<int> expected(count);
        std::iota(expected.begin(), expected.end(), 0);

        REQUIRE(computation_trace == expected);
    }

    SECTION("std::transform") {
        int count = 500;
        Generator<int, decltype(&increment)> gen(0, static_cast<size_t>(count),
                                                 increment);

        std::vector<int> two_times(static_cast<size_t>(count));
        std::transform(gen.begin(), gen.end(), two_times.begin(),
                       [](int elem) { return 2 * elem; });

        std::vector<int> expected(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i)
            expected[i] = 2 * i;

        REQUIRE(two_times == expected);
    }

    SECTION("std::equal") {
        size_t count = 500;
        Generator<int, decltype(&increment)> gen(0, count - 300, increment);
        Generator<int, decltype(&increment)> gen_sec(0, count, increment);

        REQUIRE(std::equal(gen.begin(), gen.end(), gen_sec.begin()));
        // now the first generator is consumed; they are still equal according
        // to the definition of std::equal
        REQUIRE(std::equal(gen.begin(), gen.end(), gen_sec.begin()));

        gen = Generator<int, decltype(&increment)>(20, count, increment);
        gen_sec = Generator<int, decltype(&increment)>(21, count, increment);

        // different generated values
        REQUIRE(!std::equal(gen.begin(), gen.end(), gen_sec.begin()));
    }

    SECTION("std::none_of") {
        size_t count = 500;
        Generator<int, decltype(&increment)> gen(0, count - 300, increment);

        REQUIRE(std::none_of(gen.begin(), gen.end(),
                             [](int elem) { return elem < 0; }));
    }
}
