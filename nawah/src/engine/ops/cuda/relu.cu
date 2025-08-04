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

__global__ void relu_kernel(float* c, float* t, float leakage, size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  size_t stride = gridDim.x * blockDim.x;

  for (size_t i = index; i < n; i += stride) {
    if (t[i] < 0) {
      c[i] = leakage;
    } else {
      c[i] = t[i];
    }
  }
}

Tensor CudaOps::relu(const Tensor& t, float leakage) {
  if (!t.is_contiguous()) {
    throw std::runtime_error(
        "CUDA add currently only supports contiguous tensors.");
  }

  bool c_requires_grad = t.requires_grad();
  Tensor c(t.shape(), t.dtype(), deviceToString(t.device()), c_requires_grad);

  float* c_data = static_cast<float*>(c.data_ptr().get());
  float* t_data = static_cast<float*>(t.data_ptr().get());

  size_t num_elements = t.numel();
  if (num_elements == 0) {
    return t;
  }

  int threadsPerBlock = 256;
  int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

  relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(c_data, t_data, leakage,
                                                 num_elements);

  CUDA_CHECK(cudaGetLastError());

  std::vector<__int64_t> c_shape = t.shape();
  std::vector<__int64_t> c_strides = compute_strides_(c_shape);
  bool c_requries_grad = t.requires_grad();

  auto allocator = AllocatorFactory::get(c.device());
  void* raw_ptr = allocator->allocate(num_elements);

  if (raw_ptr == nullptr) {
    throw std::runtime_error(
        "Memory allocation failed for tensor on device cuda. The device might "
        "be out of memory.");
  }

  auto deleter = [allocator](void* ptr) { allocator->deallocate(ptr); };
  c.set_data_ptr(std::shared_ptr<void>(raw_ptr, deleter));

  return c;
}
