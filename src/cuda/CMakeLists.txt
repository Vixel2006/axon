# CMakeLists.txt

cmake_minimum_required(VERSION 3.18)
project(mylib_core CXX CUDA)

# Find the CUDA toolkit
find_package(CUDA REQUIRED)

# This creates a library from your source files.
# Let's call the library "core".
add_library(core STATIC
    cuda/core/Tensor.cpp
    # Add other .cpp/.cu files here as you create them
)

# Tell CMake that our library needs CUDA features
target_link_libraries(core PRIVATE ${CUDA_LIBRARIES})

# Make sure the compiler can find your header files
target_include_directories(core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)