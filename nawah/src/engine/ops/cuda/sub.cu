#include <cuda_runtime.h>
#include <stdexcept>
#include "tensor.h"
#include "engine/ops.h"
#include "helpers.h"

#define CUDA_CHECK(err)                                                \
  do {                                                                 \
    cudaError_t err_ = (err);                                          \
    if (err_ != cudaSuccess) {                                         \
      throw std::runtime_error("CUDA Error: " +                        \
                               std::string(cudaGetErrorString(err_))); \
    }                                                                  \
  } while (0)

__global__ void sub_kernel(float* __restrict__ c, const float* __restrict__ a,
                           const float* __restrict__ b, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += gridDim.x * blockDim.x) {
        c[i] = a[i] - b[i];
    }
}


Tensor CudaOps::sub(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Tensor shapes must be identical for subtraction.");
    }

    if (a.device().type != DeviceType::CUDA || b.device().type != DeviceType::CUDA) {
        throw std::runtime_error("CUDA::sub can only operate on CUDA tensors.");
    }

    if (!a.is_contiguous() || !b.is_contiguous()) {
        throw std::runtime_error("CUDA::sub currently only supports contiguous tensors.");
    }

    Tensor c(a.shape(), a.dtype(), deviceToString(a.device()), a.requires_grad() || b.requires_grad());

    const size_t num_elements = a.numel();
    if (num_elements == 0) {
        return c;
    }

    const float* a_data = static_cast<const float*>(a.data_ptr().get());
    const float* b_data = static_cast<const float*>(b.data_ptr().get());
    float* c_data = static_cast<float*>(c.data_ptr().get());

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    sub_kernel<<<blocksPerGrid, threadsPerBlock>>>(c_data, a_data, b_data, num_elements);

    CUDA_CHECK(cudaGetLastError());

    return c;
}
