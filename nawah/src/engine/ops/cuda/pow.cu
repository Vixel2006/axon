#include <cuda_runtime.h>

#include <stdexcept>

#include "allocator/allocatorFactory.h"
#include "device.h"
#include "helpers.h"
#include "engine/ops.h"
#include "tensor.h"

#define CUDA_CHECK(err)                                                \
  {                                                                    \
    cudaError_t err_ = (err);                                          \
    if (err_ != cudaSuccess) {                                         \
      throw std::runtime_error("CUDA Error: " +                        \
                               std::string(cudaGetErrorString(err_))); \
    }                                                                  \
  }


Tensor CudaOps::pow(const Tensor& base, float exponent) {
  // TODO: pow gpu implemenation
}


Tensor CudaOps::pow(const Tensor& base, const Tensor& exponent) {
  // TODO: pow gpu implemenation
}

