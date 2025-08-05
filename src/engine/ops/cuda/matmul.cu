#include <cuda_runtime.h>
#include <stdexcept>
#include "tensor.h"
#include "engine/ops.h"
#include "helpers.h"
#include "utils.h"

#define CUDA_CHECK(err)                                                \
  do {                                                                 \
    cudaError_t err_ = (err);                                          \
    if (err_ != cudaSuccess) {                                         \
      throw std::runtime_error("CUDA Error: " +                        \
                               std::string(cudaGetErrorString(err_))); \
    }                                                                  \
  } while (0)

#define TILE_DIM 32

__global__ void batched_matmul_kernel_optimized(
    const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c,
    int M, int N, int K,
    int64_t a_batch_stride, int64_t b_batch_stride, int64_t c_batch_stride) {

    const int batch_idx = blockIdx.z;

    const float* a_batch = a + batch_idx * a_batch_stride;
    const float* b_batch = b + batch_idx * b_batch_stride;
    float* c_batch = c + batch_idx * c_batch_stride;

    __shared__ float a_tile[TILE_DIM][TILE_DIM];
    __shared__ float b_tile[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float c_value = 0.0f;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        if (row < M && (t * TILE_DIM + tx) < K) {
            a_tile[ty][tx] = a_batch[(row * K) + (t * TILE_DIM + tx)];
        } else {
            a_tile[ty][tx] = 0.0f;
        }

        if ((t * TILE_DIM + ty) < K && col < N) {
            b_tile[ty][tx] = b_batch[((t * TILE_DIM + ty) * N) + col];
        } else {
            b_tile[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            c_value += a_tile[ty][k] * b_tile[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        c_batch[row * N + col] = c_value;
    }
}


Tensor CudaOps::matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() < 2 || b.ndim() < 2) {
        throw std::runtime_error("CUDA::matmul requires at least 2 dimensions.");
    }
    if (a.shape().back() != b.shape()[b.ndim() - 2]) {
        throw std::runtime_error("Matrix inner dimensions do not match for matmul.");
    }
    if (a.device().type != DeviceType::CUDA || b.device().type != DeviceType::CUDA) {
        throw std::runtime_error("CUDA::matmul can only operate on CUDA tensors.");
    }
    if (!a.is_contiguous() || !b.is_contiguous()) {
        throw std::runtime_error("CUDA::matmul currently only supports contiguous tensors.");
    }

    std::vector<int64_t> c_shape = compute_broadcast_matmul_shape(a, b);
    Tensor c(c_shape, a.dtype(), deviceToString(a.device()), a.requires_grad() || b.requires_grad());

    Tensor a_exp = a.broadcast(c_shape);
    Tensor b_exp = b.broadcast(c_shape);

    const int c_dims = c_shape.size();
    const int M = c_shape[c_dims - 2];
    const int N = c_shape[c_dims - 1];
    const int K = a_exp.shape().back();

    int64_t batch_count = 1;
    for (size_t i = 0; i < c_dims - 2; ++i) {
        batch_count *= c_shape[i];
    }
    
    const int64_t a_batch_stride = (a_exp.ndim() > 2) ? a_exp.strides()[a_exp.ndim() - 3] : 0;
    const int64_t b_batch_stride = (b_exp.ndim() > 2) ? b_exp.strides()[b_exp.ndim() - 3] : 0;
    const int64_t c_batch_stride = (c.ndim() > 2) ? c.strides()[c.ndim() - 3] : 0;

    const float* a_data = static_cast<const float*>(a_exp.data_ptr().get());
    const float* b_data = static_cast<const float*>(b_exp.data_ptr().get());
    float* c_data = static_cast<float*>(c.data_ptr().get());

    const dim3 threadsPerBlock(TILE_DIM, TILE_DIM, 1);
    const dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                             (M + threadsPerBlock.y - 1) / threadsPerBlock.y,
                             batch_count);

    batched_matmul_kernel_optimized<<<blocksPerGrid, threadsPerBlock>>>(
        a_data, b_data, c_data, M, N, K,
        a_batch_stride, b_batch_stride, c_batch_stride);

    CUDA_CHECK(cudaGetLastError());

    return c;
}
