cmake_minimum_required(VERSION 3.10)

project(axon C CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "75") # Or a more general approach like "all" or specific architectures

find_path(SLEEF_INCLUDE_DIR sleef.h
          PATHS ${CMAKE_CURRENT_SOURCE_DIR}/vendor/sleef/build/include
          NO_DEFAULT_PATH)

find_library(SLEEF_LIBRARY NAMES sleef
             PATHS ${CMAKE_CURRENT_SOURCE_DIR}/vendor/sleef/build/lib
             NO_DEFAULT_PATH STATIC)

if(SLEEF_INCLUDE_DIR AND SLEEF_LIBRARY)
    message(STATUS "Found SLEEF include: ${SLEEF_INCLUDE_DIR}")
    message(STATUS "Found SLEEF library: ${SLEEF_LIBRARY}")
else()
    message(FATAL_ERROR "Could not find SLEEF library or include directory.")
endif()

add_library(axon SHARED
  src/tensor.c src/ops/cpu/unary_ops_cpu.c src/ops/cpu/reduction_ops_cpu.c src/ops/cpu/init_ops_cpu.c
  src/ops/cpu/binary_ops_cpu.c src/ops/cpu/binary_ops_scalar_cpu.c src/ops/cpu/movement_ops_cpu.c
  src/autograd/cpu/binary_ops_grad_cpu.c src/autograd/cpu/unary_ops_grad_cpu.c
  src/autograd/cpu/reduction_ops_grad_cpu.c src/utils.c src/autograd/autograd_utils.c 
  src/optimizers/cpu/sgd.c src/optimizers/cpu/adam.c src/optimizers/cpu/zero_grad.c src/autograd/cpu/movement_ops_grad_cpu.c
  src/ops/cuda/binary_ops_cuda.cu src/ops/cuda/init_ops_cuda.cu src/ops/cuda/binary_ops_scalar_cuda.cu
  src/ops/cuda/unary_ops_cuda.cu src/ops/cuda/reduction_ops_cuda.cu src/ops/cuda/movement_ops_cuda.cu
  src/autograd/cuda/binary_ops_grad_cuda.cu src/autograd/cuda/reduction_ops_grad_cuda.cu src/autograd/cuda/movement_ops_grad_cuda.cu
  src/autograd/cuda/unary_ops_grad_cuda.cu src/optimizers/cuda/sgd_cuda.cu src/optimizers/cuda/adam_cuda.cu src/optimizers/cuda/zero_grad_cuda.cu
)

target_include_directories(axon PUBLIC include ${SLEEF_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} ${CMAKE_BINARY_DIR})
target_link_libraries(axon PRIVATE ${SLEEF_LIBRARY} ${CUDA_LIBRARIES})
target_compile_options(axon PRIVATE -mavx2 -mfma)
set_target_properties(axon PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(axon PROPERTIES CUDA_VISIBILITY_PRESET default)

# Set output directory for the shared library
# This will place libidrak.so (or idrak.dll/idrak.dylib) in the build directory
include(GenerateExportHeader)
set(CMAKE_GENERATE_EXPORT_HEADER_LOCATION "${CMAKE_BINARY_DIR}")
generate_export_header(axon)

# set_target_properties(axon PROPERTIES CXX_VISIBILITY_PRESET hidden)
# set_target_properties(axon PROPERTIES C_VISIBILITY_PRESET hidden)
# set_target_properties(axon PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

set_target_properties(axon PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
