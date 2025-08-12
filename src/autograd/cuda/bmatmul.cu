#include "autograd/ops.h"
#include "tensor.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_DIM 32

__global__ void bmatmul_grad_A_kernel_optimized(
    const float* __restrict__ out_grad_p, const float* __restrict__ b_p, float* __restrict__ a_grad_p,
    int M, int K, int N,
    int64_t out_grad_batch_stride, int64_t b_batch_stride, int64_t a_grad_batch_stride) {
    
    const int batch_idx = blockIdx.z;

    const float* out_grad_batch = out_grad_p + batch_idx * out_grad_batch_stride;
    const float* b_batch = b_p + batch_idx * b_batch_stride;
    float* a_grad_batch = a_grad_p + batch_idx * a_grad_batch_stride;

    __shared__ float out_grad_tile[TILE_DIM][TILE_DIM];
    __shared__ float b_tile[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float grad_val = 0.0f;
    
    for (int t = 0; t < (N + TILE_DIM - 1) / TILE_DIM; ++t) {
        if (row < M && (t * TILE_DIM + tx) < N) {
            out_grad_tile[ty][tx] = out_grad_batch[row * N + (t * TILE_DIM + tx)];
        } else {
            out_grad_tile[ty][tx] = 0.0f;
        }

        int b_row = blockIdx.x * TILE_DIM + ty;
        int b_col = t * TILE_DIM + tx;
        if (b_row < K && b_col < N) {
            b_tile[ty][tx] = b_batch[b_row * N + b_col];
        } else {
            b_tile[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            grad_val += out_grad_tile[ty][k] * b_tile[tx][k];
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        atomicAdd(&a_grad_batch[row * K + col], grad_val);
    }
}

__global__ void bmatmul_grad_B_kernel_optimized(
    const float* __restrict__ a_p, const float* __restrict__ out_grad_p, float* __restrict__ b_grad_p,
    int M, int K, int N,
    int64_t a_batch_stride, int64_t out_grad_batch_stride, int64_t b_grad_batch_stride) {
    
    const int batch_idx = blockIdx.z;

    const float* a_batch = a_p + batch_idx * a_batch_stride;
    const float* out_grad_batch = out_grad_p + batch_idx * out_grad_batch_stride;
    float* b_grad_batch = b_grad_p + batch_idx * b_grad_batch_stride;

    __shared__ float a_tile[TILE_DIM][TILE_DIM];
    __shared__ float out_grad_tile[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float grad_val = 0.0f;

    for (int t = 0; t < (M + TILE_DIM - 1) / TILE_DIM; ++t) {
        int a_m_idx = t * TILE_DIM + tx;
        if (row < K && a_m_idx < M) {
            a_tile[ty][tx] = a_batch[a_m_idx * K + row];
        } else {
            a_tile[ty][tx] = 0.0f;
        }

        int out_grad_m_idx = t * TILE_DIM + ty;
        if (out_grad_m_idx < M && col < N) {
            out_grad_tile[ty][tx] = out_grad_batch[out_grad_m_idx * N + col];
        } else {
            out_grad_tile[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            grad_val += a_tile[ty][k] * out_grad_tile[k][tx];
        }
        __syncthreads();
    }

    if (row < K && col < N) {
        atomicAdd(&b_grad_batch[row * N + col], grad_val);
    }
}

void CudaAutograd::matmul(Tensor& out, std::vector<Tensor>& prev) {
    Tensor& a = prev[0];
    Tensor& b = prev[1];
    Tensor out_grad = out;

    const bool a_req_grad = a.requires_grad();
    const bool b_req_grad = b.requires_grad();

    if (!a_req_grad && !b_req_grad) {
        return;
    }

    const auto& c_shape = out_grad.shape();
    const int c_dims = c_shape.size();
    const int M = c_shape[c_dims - 2];
    const int N = c_shape[c_dims - 1];
    const int K = a.shape().back();

    const float* out_grad_p = static_cast<const float*>(out_grad.grad_ptr().get());
    const float* a_p = static_cast<const float*>(a.data_ptr().get());
    const float* b_p = static_cast<const float*>(b.data_ptr().get());
    float* a_grad_p = static_cast<float*>(a.grad_ptr().get());
    float* b_grad_p = static_cast<float*>(b.grad_ptr().get());

    if (!out_grad_p || (a_req_grad && (!a_grad_p || !b_p)) || (b_req_grad && (!b_grad_p || !a_p))) {
        throw std::runtime_error("A required pointer is null in 'matmul' backward pass (CUDA).");
    }

    int64_t batch_count = 1;
    for (size_t i = 0; i < c_dims - 2; ++i) {
        batch_count *= c_shape[i];
    }
    
    const int64_t a_batch_stride = (a.ndim() > 2) ? a.strides()[a.ndim() - 3] : M * K;
    const int64_t b_batch_stride = (b.ndim() > 2) ? b.strides()[b.ndim() - 3] : K * N;
    const int64_t out_grad_batch_stride = M * N;

    const dim3 threadsPerBlock(TILE_DIM, TILE_DIM, 1);

    if (a_req_grad) {
        const dim3 gridDimA((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                            (M + threadsPerBlock.y - 1) / threadsPerBlock.y,
                             batch_count);

        bmatmul_grad_A_kernel_optimized<<<gridDimA, threadsPerBlock>>>(
            out_grad_p, b_p, a_grad_p, M, K, N,
            out_grad_batch_stride, b_batch_stride, a_batch_stride
        );
        CUDA_CHECK(cudaGetLastError());
    }

    if (b_req_grad) {
        const dim3 gridDimB((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                            (K + threadsPerBlock.y - 1) / threadsPerBlock.y,
                             batch_count);
        
        bmatmul_grad_B_kernel_optimized<<<gridDimB, threadsPerBlock>>>(
            a_p, out_grad_p, b_grad_p, M, K, N,
            a_batch_stride, out_grad_batch_stride, b_batch_stride
        );
        CUDA_CHECK(cudaGetLastError());
    }
}
