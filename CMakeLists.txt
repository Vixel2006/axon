cmake_minimum_required(VERSION 3.10)

project(elnawah C)

add_library(elnawah SHARED src/tensor.c)

target_include_directories(elnawah PUBLIC include)

# Set output directory for the shared library
# This will place libelnawah.so (or elnawah.dll/elnawah.dylib) in the build directory
set_target_properties(elnawah PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")