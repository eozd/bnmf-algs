#!/usr/bin/env bash

# Set compiler levels to travis level
sed -i "s/-DCMAKE_CXX_COMPILER=.*/-DCMAKE_CXX_COMPILER=${CXX}/g" build.config
sed -i "s/-DCMAKE_C_COMPILER=.*/-DCMAKE_C_COMPILER=${CC}/g" build.config

# Disable OpenMP
sed -i 's/-DBUILD_OPENMP=.*/-DBUILD_OPENMP=OFF/g' build.config

# Disable CUDA
sed -i 's/-DBUILD_CUDA=.*/-DBUILD_CUDA=OFF/g' build.config
