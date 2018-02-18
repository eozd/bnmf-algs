## bnmf-algs ![build-status](https://travis-ci.org/eozd/bnmf-algs.svg?branch=master)
Nonnegative matrix factorization and matrix allocation algorithm implementations
in C++.

### Requirements
1. cmake 3.2.2 and above
2. g++ 5 and above
3. Eigen 3.2.0 and above
4. GSL 2.1 and above

### Building
To build the project type the following commands in the project root directory:
```
mkdir -p build
cd build
cmake ..
make
```

### Tests
To run the tests after building the project, type
```
./tests
```
inside the ```build``` directory.

### Documentation
To generate the doxygen documentation, run
```
doxygen doc/Doxyfile
```
in the project root directory. Then, you can view the documentation in HTML
format by opening ```doc/index.html``` using your browser.
