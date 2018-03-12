#pragma once

#include <functional>
#include <memory>

namespace bnmf_algs {
namespace util {

/**
 * @brief A template iterator that generates values via computation using the
 * previous value.
 *
 * ComputationIterator is a template class that can compute new values on-the-go
 * by storing only the most recently computed value. The computation to perform
 * at each step is taken as a constructor parameter and must be invokable
 * (functions or functors).
 *
 * After being constructed with an initial value, each call to pre-increment
 * operator (++it) would compute the next value in-place (modify the previous
 * value). Hence, prefix increment is an operation that mutates the actual
 * computation results.
 *
 * A call to dereference operator (*it) would return a const reference to the
 * computed value.
 *
 * Two ComputationIterator types can be compared using the inequality operator.
 * Two ComputationIterator types are equal if they have been advanced the same
 * number of times (number of operator++ invocations are the same).
 *
 * ComputationIterator doesn't store any of the results by value itself.
 * Instead, pointers to all the required variables are taken at construction
 * time. Then, the values pointed by these pointers are updated. This is in line
 * with pointer/iterator semantics in the sense that incrementing a copy of an
 * iterator performs a step of the computation and updates the values pointed
 * by the pointer variables. For example,
 *
 * @code
 * // Usual C++ iterators
 * std::vector<int> vec{1, 2, 3};
 * auto beg = vec.begin();
 * auto it = beg;
 * ++(*beg);
 * // *it == 2
 *
 * // ComputationIterator
 * int init = 0, beg_index = 0, end_index = 5;
 * Computer computer;
 * ComputationIterator<int, Computer> begin(&init, &beg_index, &computer);
 * ComputationIterator<int>, Computer end(&end_index);
 * auto it = beg;
 * ++beg;
 * // *it has changed as well
 * @endcode
 *
 * Since ComputationIterator provides a similar API to STL ForwardIterator,
 * it can be used with STL algorithms that only use the current value at a time
 * and doesn't depend on any of the past values. For example, to transform a
 * sequence of integers:
 *
 * @code
 * int init = 0, beg_index = 0, end_index = 5;
 * ComputationIterator<int, Computer> begin(&init, &beg_index, &computer);
 * ComputationIterator<int, Computer> end(&end_index);
 *
 * std::vector<int> out(end_index);
 * std::transform(begin, end, out.begin(), multiply_by_2);
 *
 * // out == {0, 2, 4, 6, 8}
 * @endcode
 *
 * However, it is not possible to use std::max_element since when an
 * iterator it is incremented, the value pointed by the iterator is updated and
 * all copies of it point to the updated value, as well. Fixing this issue would
 * require actually storing the past values which is not the functionality
 * provided by ComputationIterator whose main purpose is to store only the most
 * recently computed value.
 *
 * Given Computer type must have a
 * @code void operator()(size_t curr_step, T& prev_val); @endcode call operator
 * that will be called to update the previous value in-place. Example Computers
 * for ComputationIterator<std::string, Computer> may be
 * @code
 * void str_computer(size_t curr_step, std::string& prev_val) {
 *     // modify prev_val in-place to compute next value
 * }
 * @endcode
 * or
 * @code
 * class StrComputer {
 * public:
 *     void operator()(size_t curr_step, std::string& prev_val) {
 *         // modify prev_val in-place to compute next_value
 *     }
 * }
 * @endcode
 *
 * See bnmf_algs::util::Generator for an easier and more automatic API when
 * dealing with generator expressions.
 *
 * @tparam T Type of the resulting values of the computation that will be
 * performed at each step.
 * @tparam Computer Type of the invocable computer object. May be a
 * functor/std::function/function pointer and so on.
 */
template <typename T, typename Computer>
class ComputationIterator : public std::iterator<std::forward_iterator_tag, T> {
  public:
    /**
     * @brief ComputationIterator constructor that takes pointers for initial
     * value, current step count and the computer.
     *
     * The actual values pointed by init_val_ptr and step_count_ptr are updated
     * at each call to prefix increment operator (++it). If the type pointed
     * by computer_ptr is a functor and its call operator mutates the object,
     * then the computer functor is modified as well.
     *
     * Note that all the pointers must be valid before a call to any of the
     * methods of ComputationIterator. Hence, making sure that these pointers
     * are valid is up to the programmer.
     *
     * @param init_val_ptr Pointer pointing to the initial value of the
     * computation.
     * @param step_count_ptr Pointer pointing to the step count of the
     * computation.
     * @param computer_ptr Pointer pointing to the computer function/functor.
     */
    ComputationIterator(T* init_val_ptr, size_t* step_count_ptr,
                        Computer* computer_ptr)
        : curr_val_ptr(init_val_ptr), step_count_ptr(step_count_ptr),
          computer_ptr(computer_ptr){};

    /**
     * @brief ComputationIterator constructor that takes a pointer for the
     * current step only.
     *
     * If increment or dereference operators are not going to be used by this
     * iterator, giving only the current step count is enough. On the other
     * hand, if a ComputationIterator constructor using this constructor calls
     * increment or dereference operators, the resulting behaviour is undefined.
     * (Dereference nullptr)
     *
     * @param step_count_ptr Pointer pointing to the step count of the
     * computation.
     */
    explicit ComputationIterator(size_t* step_count_ptr)
        : curr_val_ptr(nullptr), step_count_ptr(step_count_ptr),
          computer_ptr(nullptr) {}

    /**
     * @brief Pre-increment operator that computes the next value of the
     * computation and updates step_count and curr_val.
     *
     * A call to operator++ computes the next value from the previous value and
     * step count using the computer function/functor. After this function
     * executes, the values pointed by step_count_ptr and curr_val_ptr is
     * updated.
     *
     * If a copy of an iterator calls operator++, then the values pointed by the
     * original are updated as well (they point to the same values). For
     * example,
     *
     * @code
     * // ComputationIterator
     * int init = 0, beg_index = 0, end_index = 5;
     * Op computer;
     * ComputationIterator<int, Op> begin(&init, &beg_index, &computer);
     * ComputationIterator<int, Op> end(&end_index);
     * auto it = beg;
     * ++beg;
     * // *it has changed as well
     * @endcode
     *
     * @return Return a reference to the current ComputationIterator object.
     */
    ComputationIterator& operator++() {
        (*computer_ptr)(*step_count_ptr, *curr_val_ptr);
        ++(*step_count_ptr);
        return *this;
    }

    /**
     * @brief Dereference operator to the get a const reference to the most
     * recently computed value.
     *
     * @return const reference to the most recently computed value.
     */
    const T& operator*() const { return *curr_val_ptr; }

    /**
     * @brief Member access operator to access the members of the most recently
     * computed value.
     *
     * @return const pointer to the most recently computed value.
     */
    const T* operator->() const { return curr_val_ptr; }

    /**
     * @brief Equality operator.
     *
     * Two ComputationIterator objects are equal if their step counts are equal.
     * The values computed during computation are not compared during equality
     * testing.
     *
     * @param other Other ComputationIterator object.
     * @return true if this ComputationIterator is equal to the other.
     */
    bool operator==(const ComputationIterator& other) const {
        return *(this->step_count_ptr) == *(other.step_count_ptr);
    }

    /**
     * @brief Inquality operator.
     *
     * Two ComputationIterator objects are not equal if their step counts are
     * not equal. The values computed during computation are not compared during
     * inequality testing.
     *
     * @param other Other ComputationIterator object.
     * @return true if this ComputationIterator is not equal to the other.
     */
    bool operator!=(const ComputationIterator& other) const {
        return !(*this == other);
    }

  private:
    T* curr_val_ptr;
    size_t* step_count_ptr;
    Computer* computer_ptr;
};

/**
 * @brief A template generator that generates values from an initial value by
 * applying a computation to the previous value a given number of times.
 *
 * A generator object is very similar to python generator expressions in that
 * it generates values from an initial value by applying a function repeatedly.
 * Only the most recently computed value is stored and returned.
 *
 * After the specified number of values are generated by repeatedly applying the
 * computer function/functor, Generator object gets consumed and cannot generate
 * any more values. Hence, to generate more values, a new Generator is required.
 * For example,
 *
 * @code
 * void increment(size_t step, int& prev) {
 *     ++prev;
 * }
 *
 * Generator<int, decltype(&increment)> gen(0, 50, increment);
 * std::vector<int> numbers;
 * for (auto num : gen) {
 *     numbers.push_back(num);
 * }
 * // numbers.size() == 50
 * // gen.begin() == gen.end()
 *
 * std::vector<int> new_numbers(numbers);
 * for (auto num : gen) {
 *     new_numbers.push_back(num);
 * }
 * // new_numbers.size() == 0
 * @endcode
 *
 * Since Generator object provides begin() and end() methods, it can be used
 * with range-based for loops as seen in the previous example. Regular
 * iterator-based for loops are also supported:
 *
 * @code
 * Op increment;
 * Generator<int, Op> gen(0, 50, increment);
 * for (auto it = gen.begin(); it != gen.end(); ++it) {
 *     numbers.push_back(num);
 * }
 * @endcode
 *
 * @tparam T Type of the values to generate.
 * @tparam Computer Type of the computer functor/function/etc. to use.
 */
template <typename T, typename Computer> class Generator {
    // todo: Can we use the return type of Computer to get rid of T?
  public:
    /**
     * @brief Iterator type that will be used to compute values repeatedly
     * without explicitly storing all the values.
     */
    using iter_type = ComputationIterator<T, Computer>;

  public:
    /**
     * @brief Generator constructor.
     *
     * Generator constructor takes an initial value and an iteration count.
     * init_val is the first value that will get generated. iter_count is the
     * number of values that will get generated. For example,
     *
     * @code
     * Generator<int, Op> gen(0, 5, increment);  // 0, 1, 2, 3, 4
     * gen = Generator<int, Op>(20, 4, decrement);  // 20, 19, 18, 17
     * @endcode
     *
     * @param init_val Initial value of the sequence to generate. This is the
     * first value that will get generated.
     * @param iter_count Number of values to generate.
     * @param computer Computer function/functor that will compute the next
     * value from the previous value. Computer is taken as an rvalue reference
     * and is moved into this Generator object.
     */
    Generator(const T& init_val, size_t iter_count, Computer&& computer)
        : init_val(init_val), curr_step_count(0), total_iter_count(iter_count),
          computer(std::move(computer)),
          begin_it(&(this->init_val), &(this->curr_step_count),
                   &(this->computer)),
          end_it(&(this->total_iter_count)){};

    /**
     * @brief Copy constructor.
     *
     * Generator class manually implements its copy constructor to correctly
     * copy the begin and end iterators which should point to the copy
     * constructed object's member variables.
     *
     * @param other Other Generator object to copy construct from.
     */
    Generator(const Generator& other)
        : init_val(other.init_val), curr_step_count(other.curr_step_count),
          total_iter_count(other.total_iter_count), computer(other.computer),
          begin_it(iter_type(&(this->init_val), &(this->curr_step_count),
                             &(this->computer))),
          end_it(iter_type(&(this->total_iter_count))) {}

    /**
     * @brief Copy assignment operator.
     *
     * Generator class manually implements its copy assignment operator to
     * correctly copy the begin and end iterators which should point to the
     * copy assigned object's member variables.
     *
     * @param other Other Generator object to copy assign from.
     *
     * @return Reference to the assigned Generator object.
     */
    Generator& operator=(const Generator& other) {
        this->init_val = other.init_val;
        this->curr_step_count = other.curr_step_count;
        this->total_iter_count = other.total_iter_count;
        this->computer = other.computer;

        this->begin_it = iter_type(&(this->init_val), &(this->curr_step_count),
                                   &(this->total_iter_count));
        this->end_it = iter_type(&(this->total_iter_count));

        return *this;
    }

    /**
     * @brief Move constructor.
     *
     * Generator class manually implements its move constructor to
     * correctly construct begin and end iterators which should point to
     * move constructed object's member variables.
     *
     * @param other Other Generator object to move from.
     */
    Generator(Generator&& other)
        : init_val(std::move(other.init_val)),
          curr_step_count(std::move(other.curr_step_count)),
          total_iter_count(std::move(other.total_iter_count)),
          computer(std::move(other.computer)),
          begin_it(iter_type(&(this->init_val), &(this->curr_step_count),
                             &(this->computer))),
          end_it(iter_type(&(this->total_iter_count))) {}

    /**
     * @brief Move assignment operator.
     *
     * Generator class manually implements its move assignment operator to
     * correctly construct begin and end iterators which should point to
     * move assigned object's member variables.
     *
     * @param other Other Generator object to move from.
     *
     * @return Reference to move assigned Generator object.
     */
    Generator& operator=(Generator&& other) {
        this->init_val = std::move(other.init_val);
        this->curr_step_count = std::move(other.curr_step_count);
        this->total_iter_count = std::move(other.total_iter_count);
        this->computer = std::move(other.computer);

        this->begin_it = iter_type(&(this->init_val), &(this->curr_step_count),
                                   &(this->computer));
        this->end_it = iter_type(&(this->total_iter_count));

        return *this;
    }
    /**
     * @brief Return the iterator pointing to the previously computed value.
     *
     * Note that since only the most recently computed value is stored, the
     * semantics for begin() is not the same STL container::begin(). After an
     * iterator is incremented, the next value is computed and the previous
     * value is forgotten.
     *
     * All the returned begin iterators point to the same values. Hence, once a
     * copy of a begin iterator is incremented and hence the pointed values are
     * updated, all the subsequent calls to begin() will return iterators
     * pointing to these updated values.
     *
     * An increment to any of the iterators update the values pointed by all the
     * iterators. Hence, once the total number of calls to increment operator
     * on iterators is equal to iter_count, all the iterators are equal to
     * Generator::end().
     *
     * Note that incrementing any of the iterators beyond Generator::end()
     * still produces new values. However, after that point, it is no longer
     * possible to make equivalence checks against Generator::end(). Hence, in
     * such a case iterators must be handled manually.
     *
     * @return Iterator pointing to the most recently computed value.
     */
    iter_type begin() { return begin_it; }

    /**
     * @brief Return the end iterator.
     *
     * End iterator is a sentinel iterator. If an iterator returned by
     * Generator::begin() is equal to Generator::end(), then this Generator
     * object is consumed and doesn't produce any values when used with
     * range-based for loops.
     *
     * @return End iterator.
     */
    iter_type end() { return end_it; }

  private:
    T init_val;
    size_t curr_step_count;
    size_t total_iter_count;
    Computer computer;
    iter_type begin_it;
    iter_type end_it;
};
} // namespace util
} // namespace bnmf_algs
