#include "engine/ops.h"
#include "tensor.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>

__global__ void fill_ones_kernel(float* data, size_t numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) data[idx] = 1.0f;
}

__global__ void fill_uniform_kernel(float* data, size_t numel, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        curandState_t state;
        curand_init(seed, idx, 0, &state);
        data[idx] = curand_uniform(&state);
    }
}

__global__ void fill_randn_kernel(float* data, size_t numel, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        curandState_t state;
        curand_init(seed, idx, 0, &state);
        data[idx] = curand_normal(&state);
    }
}


void CudaOps::fill_zeros(Tensor& t) const {
    cudaMemset(t.raw_ptr(), 0, t.numel() * sizeof(float));
}

void CudaOps::fill_ones(Tensor& t) const {
    const int threads = 256;
    const int blocks = (t.numel() + threads - 1) / threads;
    fill_ones_kernel<<<blocks, threads>>>(static_cast<float*>(t.raw_ptr()), t.numel());
}

void CudaOps::fill_uniform(Tensor& t) const {
    const int threads = 256;
    const int blocks = (t.numel() + threads - 1) / threads;
    unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    fill_uniform_kernel<<<blocks, threads>>>(static_cast<float*>(t.raw_ptr()), t.numel(), seed);
}

void CudaOps::fill_randn(Tensor& t) const {
    const int threads = 256;
    const int blocks = (t.numel() + threads - 1) / threads;
    unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    fill_randn_kernel<<<blocks, threads>>>(static_cast<float*>(t.raw_ptr()), t.numel(), seed);
}
