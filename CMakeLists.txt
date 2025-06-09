
cmake_minimum_required(VERSION 3.24)
project(h5_benchs)


add_executable(h5_bench
  main.cpp
  configuration.h
  configuration.cpp
  C_multiple_datasets.h
  C_multiple_datasets.cpp
  C_single_dataset.h
  C_single_dataset.cpp
)

# TODO: (hdf5::hdf5 (C), hdf5::hdf5_cpp (C++), hdf5::hdf_hl (C HL), hdf5::hdf_hl_cpp, HighFive)


# HDF5 library, release 1.14.6


find_package(HDF5 COMPONENTS C CXX HL NO_MODULE REQUIRED shared)

target_link_libraries(h5_bench PUBLIC hdf5::hdf5-shared)
target_compile_features(h5_bench PRIVATE cxx_std_17)


