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


Tensor CudaOps::div(const Tensor& numerator, float denominator) {
  // TODO: div gpu implemenation
}


Tensor CudaOps::div(const Tensor& numerator, const Tensor& denominator) {
  // TODO: div gpu implemenation
}
