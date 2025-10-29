cmake_minimum_required(VERSION 3.10)

project(axon C CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "86") # Set to 8.6 for NVIDIA GeForce RTX 2050

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
  src/tensor.c
  src/tensor_cpu.c
  src/tensor_cuda.cu

  # CPU Operations
  src/ops/cpu/binary/add.c
  src/ops/cpu/binary/sub.c
  src/ops/cpu/binary/mul.c
  src/ops/cpu/binary/div.c
  src/ops/cpu/binary/matmul.c
  src/ops/cpu/binary/conv2d.c
  src/ops/cpu/binary/dot.c
  src/ops/cpu/binary/pow.c

  src/ops/cpu/binary_scalar/add.c
  src/ops/cpu/binary_scalar/sub.c
  src/ops/cpu/binary_scalar/rsub.c
  src/ops/cpu/binary_scalar/mul.c
  src/ops/cpu/binary_scalar/div.c
  src/ops/cpu/binary_scalar/rdiv.c
  src/ops/cpu/binary_scalar/pow.c

  src/ops/cpu/init/zeros.c
  src/ops/cpu/init/ones.c
  src/ops/cpu/init/randn.c
  src/ops/cpu/init/uniform.c
  src/ops/cpu/init/from_data.c
  src/ops/cpu/init/borrow.c

  src/ops/cpu/movement/view.c
  src/ops/cpu/movement/unsqueeze.c
  src/ops/cpu/movement/squeeze.c
  src/ops/cpu/movement/transpose.c
  src/ops/cpu/movement/expand.c
  src/ops/cpu/movement/broadcast.c
  src/ops/cpu/movement/concat.c

  src/ops/cpu/reduction/sum.c
  src/ops/cpu/reduction/mean.c
  src/ops/cpu/reduction/max.c
  src/ops/cpu/reduction/sum_full.c
  src/ops/cpu/reduction/mean_full.c
  src/ops/cpu/reduction/max_full.c

  src/ops/cpu/unary/relu.c
  src/ops/cpu/unary/log.c
  src/ops/cpu/unary/exp.c
  src/ops/cpu/unary/neg.c
  src/ops/cpu/unary/clip.c
  src/ops/cpu/unary/abs.c

  # CUDA Operations
  src/ops/cuda/binary/add.cu
  src/ops/cuda/binary/sub.cu
  src/ops/cuda/binary/mul.cu
  src/ops/cuda/binary/div.cu
  src/ops/cuda/binary/pow.cu
  src/ops/cuda/binary/matmul.cu
  src/ops/cuda/binary/conv2d.cu
  src/ops/cuda/binary/dot.cu

  src/ops/cuda/binary_scalar/add.cu
  src/ops/cuda/binary_scalar/sub.cu
  src/ops/cuda/binary_scalar/rsub.cu
  src/ops/cuda/binary_scalar/mul.cu
  src/ops/cuda/binary_scalar/div.cu
  src/ops/cuda/binary_scalar/rdiv.cu
  src/ops/cuda/binary_scalar/pow.cu

  # src/ops/cuda/init is empty, no .cu files to add

  src/ops/cuda/movement/concat.cu

  src/ops/cuda/reduction/sum.cu
  src/ops/cuda/reduction/mean.cu
  src/ops/cuda/reduction/max.cu
  src/ops/cuda/reduction/sum_full.cu
  src/ops/cuda/reduction/mean_full.cu
  src/ops/cuda/reduction/max_full.cu

  src/ops/cuda/unary/relu.cu
  src/ops/cuda/unary/log.cu
  src/ops/cuda/unary/exp.cu
  src/ops/cuda/unary/neg.cu
  src/ops/cuda/unary/abs.cu
  src/ops/cuda/unary/clip.cu

  # Remaining files (autograd, utils, optimizers)
  src/utils.c
  src/utils/cuda/device_management.cu

  # Autograd CPU
  src/autograd/utils/autograd_utils.c
  src/autograd/cpu/binary/add.c
  src/autograd/cpu/binary/sub.c
  src/autograd/cpu/binary/rsub.c
  src/autograd/cpu/binary/mul.c
  src/autograd/cpu/binary/pow.c
  src/autograd/cpu/binary/div.c
  src/autograd/cpu/binary/rdiv.c
  src/autograd/cpu/binary/matmul.c
  src/autograd/cpu/binary/conv2d.c
  src/autograd/cpu/binary/dot.c
  src/autograd/cpu/binary/utils.c
  src/autograd/cpu/movement/concat.c
  src/autograd/cpu/reduction/sum.c
  src/autograd/cpu/reduction/mean.c
  src/autograd/cpu/reduction/max.c
  src/autograd/cpu/reduction/sum_full.c
  src/autograd/cpu/reduction/mean_full.c
  src/autograd/cpu/reduction/max_full.c
  src/autograd/cpu/unary/relu.c
  src/autograd/cpu/unary/log.c
  src/autograd/cpu/unary/exp.c
  src/autograd/cpu/unary/abs.c
  src/autograd/cpu/unary/neg.c
  src/autograd/cpu/unary/clip.c

  # Autograd CUDA
  src/autograd/cuda/binary/add.cu
  src/autograd/cuda/binary/sub.cu
  src/autograd/cuda/binary/rsub.cu
  src/autograd/cuda/binary/mul.cu
  src/autograd/cuda/binary/pow.cu
  src/autograd/cuda/binary/div.cu
  src/autograd/cuda/binary/rdiv.cu
  src/autograd/cuda/binary/matmul.cu
  src/autograd/cuda/binary/conv2d.cu
  src/autograd/cuda/binary/dot.cu
  src/autograd/cuda/movement/concat.cu
  src/autograd/cuda/reduction/sum.cu
  src/autograd/cuda/reduction/mean.cu
  src/autograd/cuda/reduction/max.cu
  src/autograd/cuda/reduction/sum_full.cu
  src/autograd/cuda/reduction/mean_full.cu
  src/autograd/cuda/reduction/max_full.cu
  src/autograd/cuda/unary/relu.cu
  src/autograd/cuda/unary/log.cu
  src/autograd/cuda/unary/exp.cu
  src/autograd/cuda/unary/abs.cu
  src/autograd/cuda/unary/neg.cu
  src/autograd/cuda/unary/clip.cu

  # Optimizers
  src/optimizers/cpu/sgd.c
  src/optimizers/cpu/adam.c
  src/optimizers/cpu/zero_grad.c
  src/optimizers/cuda/sgd_cuda.cu
  src/optimizers/cuda/adam_cuda.cu
  src/optimizers/cuda/zero_grad_cuda.cu
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
