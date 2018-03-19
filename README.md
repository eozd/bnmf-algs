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

### Building
To build the project type the following commands in the project root directory:
```
./build.sh Release
```
Afterwards, all the executables are placed inside the build directory.

### Tests
To run the tests after building the project, type
```
./tests
```
inside the ```build``` directory.

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
