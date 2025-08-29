cmake_minimum_required(VERSION 3.10)

project(elnawah C)

find_path(SLEEF_INCLUDE_DIR sleef.h
          PATHS ${CMAKE_CURRENT_SOURCE_DIR}/vendor/sleef_install/include
          NO_DEFAULT_PATH)

find_library(SLEEF_LIBRARY NAMES sleef
             PATHS ${CMAKE_CURRENT_SOURCE_DIR}/vendor/sleef_install/lib
             NO_DEFAULT_PATH)

if(SLEEF_INCLUDE_DIR AND SLEEF_LIBRARY)
    message(STATUS "Found SLEEF include: ${SLEEF_INCLUDE_DIR}")
    message(STATUS "Found SLEEF library: ${SLEEF_LIBRARY}")
else()
    message(FATAL_ERROR "Could not find SLEEF library or include directory.")
endif()

add_library(elnawah SHARED
  src/tensor.c src/node.c src/ops/cpu/unary_ops_cpu.c src/ops/cpu/reduction_ops_cpu.c
  src/ops/cpu/binary_ops_cpu.c src/ops/cpu/binary_ops_scalar_cpu.c src/ops/cpu/movement_ops_cpu.c
  src/autograd/cpu/binary_ops_grad_cpu.c src/autograd/cpu/unary_ops_grad_cpu.c
  src/autograd/cpu/reduction_ops_grad_cpu.c
)

target_include_directories(elnawah PUBLIC include ${SLEEF_INCLUDE_DIR})
target_link_libraries(elnawah PRIVATE ${SLEEF_LIBRARY})
target_compile_options(elnawah PRIVATE -mavx2 -mfma)

# Set output directory for the shared library
# This will place libelnawah.so (or elnawah.dll/elnawah.dylib) in the build directory
set_target_properties(elnawah PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
