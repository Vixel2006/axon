#include <cuda_runtime.h>

#include <stdexcept>

#include "allocator/allocatorFactory.h"
#include "device.h"
#include "helpers.h"
#include "ops/sub.h"
#include "tensor.h"

#define CUDA_CHECK(err)                                                \
  {                                                                    \
    cudaError_t err_ = (err);                                          \
    if (err_ != cudaSuccess) {                                         \
      throw std::runtime_error("CUDA Error: " +                        \
                               std::string(cudaGetErrorString(err_))); \
    }                                                                  \
  }

__global__ void sub_kernel(float* c, float* a, float* b, size_t n) {
  size_t index = blockDim.x * blockIdx.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;

  for (int i = index; i < n; i += stride) {
    c[i] = a[i] - b[i];
  }
}

Tensor sub_gpu(const Tensor& a, const Tensor& b) {
  if (a.shape().size() != b.shape().size()) {
    throw std::runtime_error(
        "the ndim of first tensor is not the same for the second one");
  }

  for (size_t i = 0; i < a.shape().size(); ++i) {
    if (a.shape()[i] != b.shape()[i]) {
      throw std::runtime_error("the tensor shapes are mismatched.");
    }
  }

  if (a.device().type != DeviceType::CUDA ||
      b.device().type != DeviceType::CUDA) {
    throw std::runtime_error("add_gpu can only operate on CUDA tensors.");
  }
  if (!a.is_contiguous() || !b.is_contiguous()) {
    throw std::runtime_error(
        "CUDA add currently only supports contiguous tensors.");
  }

  bool c_requires_grad = a.requires_grad() || b.requires_grad();
  Tensor c(a.shape(), a.dtype(), deviceToString(a.device()), c_requires_grad);

  float* c_data = static_cast<float*>(c.data_ptr().get());
  float* a_data = static_cast<float*>(a.data_ptr().get());
  float* b_data = static_cast<float*>(b.data_ptr().get());

  size_t num_elements = a.numel();
  if (num_elements == 0) {
    return a;
  }

  int threadsPerBlock = 256;
  int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

  sub_kernel<<<blocksPerGrid, threadsPerBlock>>>(c_data, a_data, b_data,
                                                 num_elements);

  CUDA_CHECK(cudaGetLastError());

  std::vector<__int64_t> c_shape = a.shape();
  std::vector<__int64_t> c_strides = compute_strides_(c_shape);
  bool c_requries_grad = a.requires_grad() || b.requires_grad();

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
