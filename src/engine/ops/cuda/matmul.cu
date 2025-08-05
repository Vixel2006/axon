#include "engine/ops.h"
#include "tensor.h"
#include "allocator/allocatorFactory.h"
#include "helpers.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <numeric>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA Error in " #call " : ") + \
                                     cudaGetErrorString(err));              \
        }                                                                   \
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
        throw std::runtime_error("CudaOps::matmul requires at least 2 dimensions.");
    }
    if (a.shape().back() != b.shape()[b.ndim() - 2]) {
        throw std::runtime_error("Matrix inner dimensions do not match for matmul.");
    }
    if (a.device().type != DeviceType::CUDA || b.device().type != DeviceType::CUDA) {
        throw std::runtime_error("CudaOps::matmul can only operate on CUDA tensors.");
    }
    if (!a.is_contiguous() || !b.is_contiguous()) {
        throw std::runtime_error("CudaOps::matmul currently only supports contiguous tensors.");
    }

    std::vector<int64_t> c_shape = compute_broadcast_matmul_shape(a, b);
    
    std::vector<int64_t> batch_shape(c_shape.begin(), c_shape.end() - 2);
    
    std::vector<int64_t> a_bcast_shape = batch_shape;
    a_bcast_shape.push_back(a.shape()[a.ndim() - 2]);
    a_bcast_shape.push_back(a.shape()[a.ndim() - 1]);

    std::vector<int64_t> b_bcast_shape = batch_shape;
    b_bcast_shape.push_back(b.shape()[b.ndim() - 2]);
    b_bcast_shape.push_back(b.shape()[b.ndim() - 1]);

    Tensor a_exp = a.broadcast(a_bcast_shape);
    Tensor b_exp = b.broadcast(b_bcast_shape);

    auto allocator = AllocatorFactory::get(a.device());
    size_t c_numel = std::accumulate(c_shape.begin(), c_shape.end(), 1LL, std::multiplies<int64_t>());
    size_t c_size_bytes = c_numel * sizeof(float);

    void* d_c_raw = allocator->allocate(c_size_bytes);
    if (!d_c_raw) {
        throw std::runtime_error("Failed to allocate CUDA memory for output tensor via AllocatorFactory.");
    }
    float* c_data = static_cast<float*>(d_c_raw);

    const float* a_data = static_cast<const float*>(a_exp.raw_ptr());
    const float* b_data = static_cast<const float*>(b_exp.raw_ptr());

    const int c_dims = c_shape.size();
    const int M = c_shape[c_dims - 2];
    const int N = c_shape[c_dims - 1];
    const int K = a_exp.shape().back();

    int64_t batch_count = 1;
    for (size_t i = 0; i < c_dims - 2; ++i) {
        batch_count *= c_shape[i];
    }
    
    const int64_t a_batch_stride = (a_exp.ndim() > 2) ? a_exp.strides()[a_exp.ndim() - 3] : M * K;
    const int64_t b_batch_stride = (b_exp.ndim() > 2) ? b_exp.strides()[b_exp.ndim() - 3] : K * N;
    const int64_t c_batch_stride = M * N;

    const dim3 threadsPerBlock(TILE_DIM, TILE_DIM, 1);
    const dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                             (M + threadsPerBlock.y - 1) / threadsPerBlock.y,
                             batch_count);

    batched_matmul_kernel_optimized<<<blocksPerGrid, threadsPerBlock>>>(
        a_data, b_data, c_data, M, N, K,
        a_batch_stride, b_batch_stride, c_batch_stride);
    CUDA_CHECK(cudaGetLastError());

    auto deleter = [allocator](void *ptr) { allocator->deallocate(ptr); };
    std::shared_ptr<void> data(d_c_raw, deleter);

    bool c_requires_grad = a.requires_grad() || b.requires_grad();
    
    return Tensor(c_shape, compute_strides_(c_shape), a.dtype(), a.device(), data, 0, c_requires_grad, nullptr, std::nullopt);
}
