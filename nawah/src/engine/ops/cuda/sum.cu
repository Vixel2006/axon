#include <cuda_runtime.h>

#include <stdexcept>

#include "allocator/allocatorFactory.h"
#include "device.h"
#include "helpers.h"
#include "engine/ops/impl/sum.h"
#include "tensor.h"

#define CUDA_CHECK(err)                                                \
  {                                                                    \
    cudaError_t err_ = (err);                                          \
    if (err_ != cudaSuccess) {                                         \
      throw std::runtime_error("CUDA Error: " +                        \
                               std::string(cudaGetErrorString(err_))); \
    }                                                                  \
  }


Tensor sum_gpu(const Tensor& a, int dim, bool keepdim) {
  // TODO: Sum gpu implemenation
}
