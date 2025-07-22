#include "ops/matmul.h"
#include "tensor.h"
#include "utils.h"
#include "helpers.h"
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void batched_matmul_kernel(
    const float* a, const float* b, float* c,
    int64_t M, int64_t N, int64_t K,
    int64_t a_stride_m, int64_t a_stride_k,
    int64_t b_stride_k, int64_t b_stride_n,
    int64_t c_stride_m, int64_t c_stride_n,
    int64_t a_batch_stride, int64_t b_batch_stride, int64_t c_batch_stride
) {
    // We get the batch index from the block index
    int batch_idx = blockIdx.z;
    const float* a_batch = a + batch_idx * a_batch_stride;
    const float* b_batch = b + batch_idx * b_batch_stride;
    float* c_batch = c + batch_idx * c_batch_stride;

    // We first define the shared memory for the tiles to reduce memory access operations
    __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE][TILE_SIZE];

    // Here we define the thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // We get the row and col indices for the tile
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    // Now we load the tiles from the global memory
    for (int i = 0 ; i < TILE_SIZE; i += TILE_SIZE) {
        a_tile[ty + i][tx] = a[row * a_stride_m + i * a_stride_k + tx];
        b_tile[ty][tx + i] = b[col * b_stride_k + i * b_stride_n + ty];
    }

    // Now we can do the matmul
    float sum = 0.0f;
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += a_tile[ty][k] * b_tile[k][tx];
    }

    // Now we can store the result in the global memory
    if (row < M && col < N) {
        c[row * c_stride_m + col * c_stride_n] = sum;
    }
    __syncthreads();
}

Tensor matmul_gpu(const Tensor& a, const Tensor& b) {
    if (a.ndim() < 2 || b.ndim() < 2) {
        throw std::runtime_error("Matmul requires at least 2 dimensions.");
    }

    const int64_t K_a = a.shape().back();

    const int64_t K_b = b.shape()[b.ndim() - 2];

    if (K_a != K_b) {
        throw std::runtime_error("Inner dimensions for matmul do not match.");
    }

    // 2. Compute Output Shape via Broadcasting
    std::vector<int64_t> c_shape = compute_broadcast_matmul_shape(a, b);
    Tensor c = Tensor(c_shape, a.dtype(), deviceToString(a.device()));

    const int c_dims = c.shape().size();
    const int64_t M = c.shape()[c.ndim() - 2];
    const int64_t N = c.shape()[c.ndim() - 1];
    const int64_t K = K_a;

    std::vector<int64_t> batch_shape(c_shape.begin(), c_shape.end() - 2);

    // Target shape for 'a' is (batch_dims..., M, K)
    std::vector<int64_t> a_target_shape = batch_shape;

    a_target_shape.push_back(M);
    a_target_shape.push_back(K);

    // Target shape for 'b' is (batch_dims..., K, N)
    std::vector<int64_t> b_target_shape = batch_shape;

    b_target_shape.push_back(K);
    b_target_shape.push_back(N);
    
    // Use the .broadcast() method which is designed for this.
    Tensor a_exp = a.broadcast(a_target_shape);
    Tensor b_exp = b.broadcast(b_target_shape);

    int64_t batch_count = 1;

    for (size_t i = 0; i < c_dims - 2; ++i) {
        batch_count *= c.shape()[i];
    }

    int64_t a_batch_stride = a_exp.strides()[0]; // or product of dims after batch
    int64_t b_batch_stride = b_exp.strides()[0];
    int64_t c_batch_stride = c.strides()[0];

    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, batch_count);
    dim3 block(TILE_SIZE, TILE_SIZE);

    batched_matmul_kernel<<<grid, block>>>(
        static_cast<const float*>(a_exp.raw_ptr()), static_cast<const float*>(b_exp.raw_ptr()), static_cast<float*>(c.raw_ptr()),
        M, N, K,
        a_exp.strides()[a_exp.ndim() - 2], a_exp.strides()[a_exp.ndim() - 1],
        b_exp.strides()[b_exp.ndim() - 2], b_exp.strides()[b_exp.ndim() - 1],
        c.strides()[c.ndim() - 2], c.strides()[c.ndim() - 1],
        a_batch_stride, b_batch_stride, c_batch_stride
    );

    return c;
}
