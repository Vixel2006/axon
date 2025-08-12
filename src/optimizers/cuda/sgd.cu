#include "tensor.h"
#include "optimizers.h"
#include <cuda_runtime.h>

__global__ void sgd_kernel(float** params, float** grad, const size_t* sizes, int num_tensors, float lr) {
  int t = blockIdx.y;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < num_tensors && idx < sizes[t]) {
    params[t][idx] -= lr * grad[t][idx];
  }
}

void CudaOptimizers::SGD(std::vector<std::shared_ptr<Tensor>>& params, float lr) {
  int num_tensors = params.size();

  std::vector<float*> h_data_ptrs(num_tensors);
  std::vector<float*> h_grad_ptrs(num_tensors);
  std::vector<size_t> h_sizes(num_tensors);

  for (int i = 0; i < num_tensors; ++i) {
    h_data_ptrs[i] = static_cast<float*>(params[i]->data_ptr().get());
    h_grad_ptrs[i] = static_cast<float*>(params[i]->grad_ptr().get());
    h_sizes[i] = params[i]->numel();
  }

  float** d_data_ptrs;
  float** d_grad_ptrs;
  size_t* d_sizes;

  cudaMalloc(&d_data_ptrs, num_tensors * sizeof(float*));
  cudaMalloc(&d_grad_ptrs, num_tensors * sizeof(float*));
  cudaMalloc(&d_sizes, num_tensors * sizeof(size_t));

  cudaMemcpy(d_data_ptrs, h_data_ptrs.data(), num_tensors * sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy(d_grad_ptrs, h_grad_ptrs.data(), num_tensors * sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sizes, h_sizes.data(), num_tensors * sizeof(float*), cudaMemcpyHostToDevice);

  size_t max_size = *std::max_element(h_sizes.begin(), h_sizes.end());

  dim3 threads(256);
  dim3 blocks((max_size + threads.x - 1) / threads.x, num_tensors);

  sgd_kernel<<<blocks, threads>>>(d_data_ptrs, d_grad_ptrs, d_sizes, num_tensors, lr);

  cudaFree(d_data_ptrs);
  cudaFree(d_grad_ptrs);
  cudaFree(d_sizes);
}

