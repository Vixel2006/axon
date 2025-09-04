cmake_minimum_required(VERSION 3.10)

project(idrak C)

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

add_library(idrak SHARED
  src/tensor.c src/node.c src/ops/cpu/unary_ops_cpu.c src/ops/cpu/reduction_ops_cpu.c
  src/ops/cpu/binary_ops_cpu.c src/ops/cpu/binary_ops_scalar_cpu.c src/ops/cpu/movement_ops_cpu.c
  src/autograd/cpu/binary_ops_grad_cpu.c src/autograd/cpu/unary_ops_grad_cpu.c
  src/autograd/cpu/reduction_ops_grad_cpu.c src/utils.c src/autograd/autograd_utils.c src/ops/ops_utils.c
  src/optimizers/sgd.c src/optimizers/adam.c src/optimizers/zero_grad.c
)

target_include_directories(idrak PUBLIC include ${SLEEF_INCLUDE_DIR})
target_link_libraries(idrak PRIVATE ${SLEEF_LIBRARY})
target_compile_options(idrak PRIVATE -mavx2 -mfma)

# Set output directory for the shared library
# This will place libidrak.so (or idrak.dll/idrak.dylib) in the build directory
set_target_properties(idrak PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
