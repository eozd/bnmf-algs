#include <celero/Celero.h>

CELERO_MAIN

/**
 * This file is the source file for becnhmarks. Each benhmark should be
 * defined on its own header. Then, the benchmark header must be included
 * as seen below. This will automatically add the benchmark group to the list
 * of all available benchmark groups.
 */

/*********************** INCLUDE BENCHMARK HEADER BELOW ***********************/

#include "bld_add_benchmark.hpp"
#include "bld_appr_benchmark.hpp"
#include "bld_mult_benchmark.hpp"
#include "collapsed_gibbs_benchmark.hpp"
#include "collapsed_icm_benchmark.hpp"
#include "seq_greedy_bld_benchmark.hpp"
#include "nmf_benchmark.hpp"
#include "nmf_seq_greedy_bld_matrix_elems_benchmark.hpp"
#include "psi_appr_benchmark.hpp"
#include "swimmer_benchmark.hpp"

/*********************** INCLUDE BENCHMARK HEADER ABOVE ***********************/
