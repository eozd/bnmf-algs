#pragma once
#include <cstddef>

/**
 * @brief Simple counter class that counts the number of invocations of default
 * constructor, copy ctor/assign operators, move ctor/assign operators,
 * destructor
 */
template <typename T> struct Counter {
    static size_t TotalDefaultCtorCount;
    static size_t TotalCopyCount;
    static size_t TotalMoveCount;
    static size_t TotalDtorCount;

    static void reset_counters() {
        TotalDefaultCtorCount = 0;
        TotalCopyCount = 0;
        TotalMoveCount = 0;
        TotalDtorCount = 0;
    }

    Counter() : val(T()) { ++TotalDefaultCtorCount; }

    Counter(const Counter& other) {
        this->val = other.val;
        ++TotalCopyCount;
    }

    Counter& operator=(const Counter& other) {
        this->val = other.val;
        ++TotalCopyCount;
        return *this;
    }

    Counter(Counter&& other) {
        this->val = other.val;
        ++TotalMoveCount;
    }

    Counter& operator=(Counter&& other) {
        this->val = other.val;
        ++TotalMoveCount;
    }

    ~Counter() { ++TotalDtorCount; }

    T val;
};

template <typename T> size_t Counter<T>::TotalDefaultCtorCount = 0;

template <typename T> size_t Counter<T>::TotalCopyCount = 0;

template <typename T> size_t Counter<T>::TotalMoveCount = 0;

template <typename T> size_t Counter<T>::TotalDtorCount = 0;
