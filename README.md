## bnmf-algs
[![Travis Status](https://travis-ci.org/eozd/bnmf-algs.svg?branch=master)](https://travis-ci.org/eozd/bnmf-algs)
[![codecov](https://codecov.io/gh/eozd/bnmf-algs/branch/master/graph/badge.svg)](https://codecov.io/gh/eozd/bnmf-algs)
[![Documentation Status](https://readthedocs.org/projects/bnmf-algs/badge/?version=latest)](http://bnmf-algs.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Nonnegative matrix factorization and matrix allocation algorithm implementations
in C++.

### Requirements
1. cmake 3.2.2 and above
2. g++ 5 and above
3. Eigen 3.3.0 and above
4. GSL 2.1 and above
5. [Celero](https://github.com/DigitalInBlue/Celero) 2.2.0 and above (**optional**)

### Building
To build the project type the following commands in the project root directory:
```
./build.sh release
```
Afterwards, bnmf-algs shared library **libbnmf_algs.so** should be placed in build directory.

#### Clean
You can clean the built library, tests and benchmarks by running
```
./build.sh clean
```
This command removes all cmake related folders/files and all previously built targets.

### Tests
To build the tests run:
```
./build.sh release test
```
To run the tests after building, type
```
./build/tests
```

### Benchmarks
To build the benchmarks Celero library must be installed on your system (see Requirements section). After installing Celero, build the benchmarks by running:
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

### Documentation

#### Building
To generate the doxygen documentation, run
```
doxygen doc/Doxyfile
```
in the project root directory. Then, you can view the documentation in HTML
format by opening ```doc/build/html/index.html``` using your browser.

#### View Online
You can also view the documentation [online](http://bnmf-algs.readthedocs.io/en/latest/?badge=latest).
Online documentation is automatically built from the latest commit to the master branch.
