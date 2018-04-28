# bnmf-algs
[![Travis Status](https://travis-ci.org/eozd/bnmf-algs.svg?branch=master)](https://travis-ci.org/eozd/bnmf-algs)
[![codecov](https://codecov.io/gh/eozd/bnmf-algs/branch/master/graph/badge.svg)](https://codecov.io/gh/eozd/bnmf-algs)
[![Documentation Status](https://readthedocs.org/projects/bnmf-algs/badge/?version=latest)](http://bnmf-algs.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

bnmf-algs is a header only (except benchmark and CUDA parts) C++14 library
providing parallelized implementations of nonnegative matrix factorization and
allocation model algorithms [[1]](#refs). bnmf-algs is built on top of
[Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page), and uses Eigen
matrix and tensor types in its API and internal computations.  The library can
be built with OpenMP support to enable parallel implementations of algorithms.
Additionally, bnmf-algs supports CUDA 8 and above for computing large
matrix/tensor operations on an NVIDIA GPU for additional speed-up.

## Introduction
Allocation model is proposed by M. B. Kurutmaz, A. T. Cemgil, U. Şimşekli, and
S. Yıldırım (see [[1]](#refs)).  The model has a close relationship with NMF,
and admits similar inference algorithms. The problem of finding the optimal
solution to an allocation model problem is named as the Best Latent
Decomposition problem in [[1]](#refs).  This repository contains parallelized
versions of BLD (best latent decomposition) algorithms. All the theoretical
ideas behind the implemented BLD solutions are obtained from personal
communication with M. B. Kurutmaz and A. T. Cemgil.

All the implemented algorithms/functions/classes are extensively documented,
tested using edge-case and correctness tests, and benchmarked. You can easily
reach the test results on Travis and documentation on readthedocs by following
the links at the top of this page. Benchmark codes are also provided together
with this library. To run the benchmarks, you need to first compile them.
Instructions for doing so are mentioned in [build](#build) section.

Below we introduce the namespaces and main algorithms provided by bnmf-algs.

### General View
The library is divided into sections using C++ namespaces. The main namespace of
the library is bnmf-algs. The rest of the large namespaces are as follows:

#### alloc_model
This namespace contains allocation model related parameter classes and
functions.  Currently, a class to store model parameters is provided.
Additionally, functions for sampling prior matrices and tensors, and
calculating the log marginal value of a sample according to allocation model
is provided.

#### bld
This is the namespace that contains various BLD solver algorithms. Each
algorithm has a similar signature, taking the matrix to allocate into the model
tensor according to the given model parameters and various algorithm parameters
such as iteration count and epsilon values. In order to read algorithmic
descriptions, please refer to bnmf-algs library documentation.

#### cuda
cuda namespace contains utility functions and classes that make it easier to
use CUDA from C++ code.

#### nmf
nmf namespace contains a parallelized implementation of nonnegative matrix
factorization algorithm implemented according to beta divergence. The algorithm
is parallelized by using parallel Eigen matrix multiplications.

#### util
util namespace contains various utility functions and classes.

## Requirements
1. cmake 3.2.2 and above
2. g++ 5 and above
  * If bnmf-algs is built with CUDA support, then the compiler version must be
  compatible with CUDA version. For example, the newest g++ version that is
  compatible with CUDA 8 is g++ 5.
3. Eigen 3.3.0 and above
4. GSL 2.1 and above

## Optional Requirements
1. [Celero](https://github.com/DigitalInBlue/Celero) 2.2.0 and above to run
benchmark codes
2. OpenMP for multithreaded parallel execution of certain operations on
separate CPU cores
3. CUDA 8 and above to run large matrix/tensor operations on an NVIDIA GPU

## <a name="build">Building</a>
To use bnmf-algs in your project, you can follow the steps described in this
section. You can check the example programs compiled with/without CUDA support
to have an idea of how to use bnmf-algs in your project. Additionally, we
describe the main build script used to build benchmarks and tests.

### Example C++ Programs
Two example programs are provided in example and example_with_cuda directories.
They provide the same functionality, i.e. read a matrix from a given file,
factorize it, and write the resulting matrices/tensors to predefined files.
The only difference is that one of them is built using CUDA and thus offers
CUDA support whereas the other only uses OpenMP for parallelizations. Example
CMakeLists.txt files are provided in both directory to give an example
CMakeLists.txt needed to use bnmf-algs library in your project.

You can easily build both of the example projects by running the provided build
script.

#### example_with_cuda/CMakeLists.txt
This is the file that used to generate the makefile to compile the main file in
example_with_cuda directory. A brief explanation of certain parts is as follows:

1. In add_definitions part we add the necessary definitions used by bnmf-algs.
In particular, **-DUSE_OPENMP** and **-DUSE_CUDA** directives enable bnmf-algs
OpenMP and CUDA parts of bnmf-algs, respectively. The following definitions
are required by Eigen library.
2. Next, we add bnmf-algs headers into include path using include_directories
3. Required libraries are found using find_package commands. These commands
use CMake modules which are stored in cmake directory under project root. You
may need some of these files to find GSL, Eigen or CUDA libraries automatically
using CMake instead of hardcoding the library paths.
4. CUDA source files that need to be compiled using nvcc are stored in
CUDA_KERNEL_FILES constant.
5. After setting CUDA options, we finally build the executable using
CUDA_ADD_EXECUTABLE and link it to GSL and CUDA runtime.

#### Without CUDA
Building the library without CUDA support is much simpler as you don't need to
separately compile any part of bnmf-algs library. Simply including it and
linking againts GSL is enough. Explanation of CMakeLists.txt is as follows:

1. We again specify directives to be used by bnmf-algs and Eigen using
add_definitions commands.
2. bnmf-algs headers directory is specified using include_directories.
3. Eigen and GSL libraries are found using find_package.
4. Finally, executable is built with the only additional step of linking it
against the GSL runtime by target_link_libraries.

### Benchmarks and Tests
Benchmarks and tests can be built using the main build script provided in
build.sh file under project root. To configure the build process, you need to
edit build.config file. This file contains CMake flags to pass to cmake
executable. Each option is described in the comments above it. For example,
to enable OpenMP support, set
```
-DBUILD_OPENMP=ON
```
Similarly, to enable CUDA, set

```
-DBUILD_CUDA=ON
```

#### Tests
After configuring the build process, to build the tests run:
```
./build.sh release test
```
This command builds the tests in release mode. To run the tests after building,
type
```
./build/tests
```

#### Benchmarks
In order to build the benchmarks, Celero library must be installed on your
system. After installing Celero, you need to specify its include and link paths
in build.config file. Finally, you can build the benchmarks using the following
command:
```
./build.sh release bench
```
This will create a benchmark executable in ```build/benchmark```.

To see the list of benchmarks, type
```
./build/benchmark -l
```

To run a specific benchmark with name X, type
```
./build/benchmark -g X
```

#### Documentation
To generate the doxygen documentation, run
```
./build.sh doc
```
in the project root directory. Then, you can view the documentation in HTML
format by opening ```doc/build/html/index.html``` using your browser.

You can also view the documentation [online](http://bnmf-algs.readthedocs.io/en/latest/?badge=latest).
Online documentation is automatically built from the latest commit to the master branch.

#### Clean
You can clean the built tests, benchmarks and documentation by running
```
./build.sh clean
```
This command removes all cmake related folders/files and all previously built
targets.

## Known Bugs
1. When building the tests/benchmarks with CUDA support, build script needs to
be run twice consecutively since the first CUDA compilation fails. However, the
library builds without any errors in the second build.

## <a name="refs">References</a>
* [1] M. B. Kurutmaz, A. T. Cemgil, U. Şimşekli, and S. Yıldırım, “Bayesian nonnegative matrix factorization as an allocation model”
