#include <iostream>
#include <memory>
#include "util/generator.hpp"
#include "../catch2.hpp"
#include "defs.hpp"
#include <algorithm>

using namespace bnmf_algs::util;

void str_computer(size_t index, std::string& prev) {
    prev += "abc";
}

struct C {
    int val;

    C() : val(0) {
        std::cout << "default ctor" << std::endl;
    }
};

void c_computer(size_t index, C& prev) {
    prev.val++;
}

struct Op {
    void operator()(size_t index, C& prev) {
        prev.val++;
    }
};

TEST_CASE("Test generator", "[generator]") {
    //Generator<std::string> r("beg", 5, str_computer);


    Generator<C> gen(C(), 500, Op());
    for (auto it = gen.prev(); it != gen.end(); ++it) {
        std::cout << it->val << std::endl;
    }

    gen = Generator<C>(C(), 500, c_computer);
    std::cout << (gen.prev() == gen.end()) << std::endl;
    for (auto it = gen.prev(); it != gen.end(); ++it) {
        std::cout << it->val << std::endl;
    }

    // todo: test
}

