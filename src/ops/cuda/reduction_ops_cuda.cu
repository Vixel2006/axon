#include "logger.h"
#include "ops/reduction_ops.h"
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define CHECK_CUDA()                                                                               \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = cudaGetLastError();                                                      \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            LOG_ERROR("CUDA runtime error at %s:%d: %s", __FILE__, __LINE__,                       \
                      cudaGetErrorString(err));                                                    \
            return;                                                                                \
        }                                                                                          \
    } while (0)

template <int block_size>
__device__ __forceinline__ void wrap_sum_reduce(volatile float* rdata, int tid)
{
    if (block_size >= 64) rdata[tid] += rdata[tid + 32];
    if (block_size >= 32) rdata[tid] += rdata[tid + 16];
    if (block_size >= 16) rdata[tid] += rdata[tid + 8];
    if (block_size >= 8) rdata[tid] += rdata[tid + 4];
    if (block_size >= 4) rdata[tid] += rdata[tid + 2];
    if (block_size >= 2) rdata[tid] += rdata[tid + 1];
}

template <int block_size>
__global__ void sum_kernel(float* a, float* out, int n, int axis_dim, int outer_dim, int inner_dim)
{
    extern __shared__ float rdata[];
    int tid = threadIdx.x;
    int outer_idx = blockIdx.x;
    int inner_idx = blockIdx.y;

    rdata[tid] = 0.0f;
    __syncthreads();

    for (int i = tid; i < axis_dim; i += block_size)
    {
        int current_index = outer_idx * (axis_dim * inner_dim) + i * inner_dim + inner_idx;
        if (current_index < n)
        {
            rdata[tid] += a[current_index];
        }
    }
    __syncthreads();

    if (block_size >= 512)
    {
        if (tid < 256)
        {
            rdata[tid] += rdata[tid + 256];
        }
        __syncthreads();
    }

    if (block_size >= 256)
    {
        if (tid < 128)
        {
            rdata[tid] += rdata[tid + 128];
        }
        __syncthreads();
    }

    if (block_size >= 128)
    {
        if (tid < 64)
        {
            rdata[tid] += rdata[tid + 64];
        }
        __syncthreads();
    }

    if (block_size >= 64) wrap_sum_reduce<block_size>(rdata, tid);

    if (tid == 0)
    {
        int output_idx = outer_idx * inner_dim + inner_idx;
        out[output_idx] = rdata[0];
    }
}

template <int block_size>
__global__ void mean_kernel(float* a, float* out, int n, int axis_dim, int outer_dim, int inner_dim)
{
    extern __shared__ float rdata[];
    int tid = threadIdx.x;
    int outer_idx = blockIdx.x;
    int inner_idx = blockIdx.y;

    rdata[tid] = 0.0f;
    __syncthreads();

    for (int i = tid; i < axis_dim; i += 2 * block_size)
    {
        int left_index = outer_idx * axis_dim * inner_dim + i * inner_dim + inner_idx;
        int right_index =
            outer_idx * axis_dim * inner_dim + (i + block_size) * inner_dim + inner_idx;
        float left_val = left_index < n ? a[left_index] : 0.0f;
        float right_val = right_index < n ? a[right_index] : 0.0f;
        rdata[tid] += left_val + right_val;
    }
    __syncthreads();

    if (block_size >= 512)
    {
        if (tid < 256)
        {
            rdata[tid] += rdata[tid + 256];
        }
        __syncthreads();
    }

    if (block_size >= 256)
    {
        if (tid < 128)
        {
            rdata[tid] += rdata[tid + 128];
        }
        __syncthreads();
    }

    if (block_size >= 128)
    {
        if (tid < 64)
        {
            rdata[tid] += rdata[tid + 64];
        }
        __syncthreads();
    }

    if (block_size >= 64) wrap_sum_reduce<block_size>(rdata, tid);

    if (tid == 0)
    {
        int output_idx = outer_idx * inner_dim + inner_idx;
        out[output_idx] = rdata[0] / axis_dim;
    }
}

template <int block_size>
__device__ __forceinline__ void wrap_max_reduce(volatile float* rdata, int tid)
{
    if (block_size >= 64) rdata[tid] = fmaxf(rdata[tid], rdata[tid + 32]);
    if (block_size >= 32) rdata[tid] = fmaxf(rdata[tid], rdata[tid + 16]);
    if (block_size >= 16) rdata[tid] = fmaxf(rdata[tid], rdata[tid + 8]);
    if (block_size >= 8) rdata[tid] = fmaxf(rdata[tid], rdata[tid + 4]);
    if (block_size >= 4) rdata[tid] = fmaxf(rdata[tid], rdata[tid + 2]);
    if (block_size >= 2) rdata[tid] = fmaxf(rdata[tid], rdata[tid + 1]);
}

template <int block_size>
__global__ void max_kernel(float* a, float* out, int n, int axis_dim, int inner_dim, int outer_dim)
{
    extern __shared__ float rdata[];
    int tid = threadIdx.x;
    int outer_idx = blockIdx.x;
    int inner_idx = blockIdx.y;

    rdata[tid] = -FLT_MAX;
    __syncthreads();

    for (int i = tid; i < axis_dim; i += 2 * block_size)
    {
        int left_index = outer_idx * axis_dim * inner_dim + i * inner_dim + inner_idx;
        int right_index =
            outer_idx * axis_dim * inner_dim + (i + block_size) * inner_dim + inner_idx;
        float left_val = left_index < n ? a[left_index] : -FLT_MAX;
        float right_val = right_index < n ? a[right_index] : -FLT_MAX;
        rdata[tid] = fmaxf(rdata[tid], fmaxf(left_val, right_val));
    }

    if (block_size >= 512)
    {
        if (tid < 256)
        {
            rdata[tid] = fmaxf(rdata[tid], rdata[tid + 256]);
        }
    }

    if (block_size >= 256)
    {
        if (tid < 128)
        {
            rdata[tid] = fmaxf(rdata[tid], rdata[tid + 128]);
        }
    }

    if (block_size >= 128)
    {
        if (tid < 64)
        {
            rdata[tid] = fmaxf(rdata[tid], rdata[tid + 64]);
        }
    }

    if (block_size >= 64) wrap_max_reduce<block_size>(rdata, tid);

    if (tid == 0)
    {
        int output_idx = outer_idx * inner_dim + inner_idx;
        out[output_idx] = fmaxf(out[output_idx], rdata[0]);
    }
}

template <int block_size> __global__ void full_sum_kernel(const float* a, float* out, int n)
{
    extern __shared__ float rdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (block_size * 2) + threadIdx.x;
    int grid_size = blockDim.x * 2 * gridDim.x;
    rdata[tid] = 0;

    while (idx < n)
    {
        rdata[tid] += a[idx] + a[idx + block_size];
        idx += grid_size;
    }
    __syncthreads();

    if (block_size >= 512)
    {
        if (tid < 256)
        {
            rdata[tid] += rdata[tid + 256];
        }
        __syncthreads();
    }

    if (block_size >= 256)
    {
        if (tid < 128)
        {
            rdata[tid] += rdata[tid + 128];
        }
        __syncthreads();
    }

    if (block_size >= 128)
    {
        if (tid < 64)
        {
            rdata[tid] += rdata[tid + 64];
        }
        __syncthreads();
    }

    if (block_size >= 64) wrap_sum_reduce<block_size>(rdata, tid);

    if (tid == 0) out[blockIdx.x] = rdata[0];
}

template <int block_size> __global__ void full_max_kernel(float* a, float* out, int n)
{
    extern __shared__ float rdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (block_size * 2) + threadIdx.x;
    int grid_size = blockDim.x * 2 * gridDim.x;
    rdata[tid] = -FLT_MAX;
    __syncthreads();

    while (idx < n)
    {
        float val1 = a[idx];
        float val2 = (idx + block_size < n) ? a[idx + block_size] : -FLT_MAX;
        rdata[tid] = fmaxf(rdata[tid], fmaxf(val1, val2));
        idx += grid_size;
    }
    __syncthreads();

    if (block_size >= 512)
    {
        if (tid < 256)
        {
            rdata[tid] = fmaxf(rdata[tid], rdata[tid + 256]);
        }
        __syncthreads();
    }

    if (block_size >= 256)
    {
        if (tid < 128)
        {
            rdata[tid] = fmaxf(rdata[tid], rdata[tid + 128]);
        }
        __syncthreads();
    }

    if (block_size >= 128)
    {
        if (tid < 64)
        {
            rdata[tid] = fmaxf(rdata[tid], rdata[tid + 64]);
        }
        __syncthreads();
    }

    if (block_size >= 64) wrap_max_reduce<block_size>(rdata, tid);

    if (tid == 0) out[blockIdx.x] = rdata[0];
}

void sum_op_cuda(Tensor* a, Tensor* out, int axis, bool keepdim)
{
    LOG_INFO("Sum operation on cuda starting......");

    int N = numel(a->shape, a->ndim);

    if (axis < 0 || axis >= a->ndim)
    {
        LOG_ERROR("sum_op_cuda: Axis %d is out of bounds for tensor with %d dimensions.", axis,
                  a->ndim);
        return;
    }

    int outer_dim = 1;
    for (int i = 0; i < axis; ++i)
    {
        outer_dim *= a->shape[i];
    }

    int axis_dim = a->shape[axis];

    int inner_dim = 1;
    for (int i = axis + 1; i < a->ndim; ++i)
    {
        inner_dim *= a->shape[i];
    }

    int num_threads_per_block = 256;
    dim3 grid_dims(outer_dim, inner_dim);

    int output_numel = outer_dim * inner_dim;

    out->data = (Storage*) malloc(sizeof(Storage));
    if (!out->data)
    {
        LOG_ERROR("Failed to allocate Storage for out tensor in sum_op_cuda");
        return;
    }
    out->data->counter = 1;
    out->data->size = output_numel;

    cudaError_t err = cudaMalloc((void**) &out->data->data, out->data->size * sizeof(float));
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to allocate CUDA memory for out->data->data in sum_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data, free);
        return;
    }

    sum_kernel<256><<<grid_dims, num_threads_per_block, num_threads_per_block * sizeof(float)>>>(
        a->data->data, out->data->data, N, axis_dim, outer_dim, inner_dim);

    CHECK_CUDA();
    LOG_INFO("Sum operation on cuda done.");
}

void mean_op_cuda(Tensor* a, Tensor* out, int axis, bool keepdim)
{
    LOG_INFO("mean operation on cuda starting......");

    int N = numel(a->shape, a->ndim);

    if (axis < 0 || axis >= a->ndim)
    {
        LOG_ERROR("mean_op_cuda: Axis %d is out of bounds for tensor with %d dimensions.", axis,
                  a->ndim);
        return;
    }

    int outer_dim = 1;
    for (int i = 0; i < axis; ++i)
    {
        outer_dim *= a->shape[i];
    }

    int axis_dim = a->shape[axis];

    int inner_dim = 1;
    for (int i = axis + 1; i < a->ndim; ++i)
    {
        inner_dim *= a->shape[i];
    }

    int num_threads_per_block = 256;
    dim3 grid_dims(outer_dim, inner_dim);

    int output_numel = outer_dim * inner_dim;

    out->data = (Storage*) malloc(sizeof(Storage));
    if (!out->data)
    {
        LOG_ERROR("Failed to allocate Storage for out tensor in mean_op_cuda");
        return;
    }
    out->data->counter = 1;
    out->data->size = output_numel;

    cudaError_t err = cudaMalloc((void**) &out->data->data, out->data->size * sizeof(float));
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to allocate CUDA memory for out->data->data in mean_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data, free);
        return;
    }

    mean_kernel<256><<<grid_dims, num_threads_per_block, num_threads_per_block * sizeof(float)>>>(
        a->data->data, out->data->data, N, axis_dim, outer_dim, inner_dim);

    CHECK_CUDA();
    LOG_INFO("mean operation on cuda done.");
}

void max_op_cuda(Tensor* a, Tensor* out, int axis, bool keepdim)
{
    LOG_INFO("max operation on cuda starting......");

    int N = numel(a->shape, a->ndim);

    if (axis < 0 || axis >= a->ndim)
    {
        LOG_ERROR("max_op_cuda: Axis %d is out of bounds for tensor with %d dimensions.", axis,
                  a->ndim);
        return;
    }

    int outer_dim = 1;
    for (int i = 0; i < axis; ++i)
    {
        outer_dim *= a->shape[i];
    }

    int axis_dim = a->shape[axis];

    int inner_dim = 1;
    for (int i = axis + 1; i < a->ndim; ++i)
    {
        inner_dim *= a->shape[i];
    }

    int num_threads_per_block = 256;
    dim3 grid_dims(outer_dim, inner_dim);

    int output_numel = outer_dim * inner_dim;

    out->data = (Storage*) malloc(sizeof(Storage));
    if (!out->data)
    {
        LOG_ERROR("Failed to allocate Storage for out tensor in max_op_cuda");
        return;
    }
    out->data->counter = 1;
    out->data->size = output_numel;

    cudaError_t err = cudaMalloc((void**) &out->data->data, out->data->size * sizeof(float));
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to allocate CUDA memory for out->data->data in max_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data, free);
        return;
    }

    max_kernel<256><<<grid_dims, num_threads_per_block, num_threads_per_block * sizeof(float)>>>(
        a->data->data, out->data->data, N, axis_dim, inner_dim, outer_dim);

    CHECK_CUDA();
    LOG_INFO("max operation on cuda done.");
}

void sum_full_op_cuda(Tensor* a, Tensor* out)
{
    LOG_INFO("Sum operation on cuda starting......");
    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int blocks = (N + (num_threads_per_block * 2) - 1) / (num_threads_per_block * 2);

    float* h_out_partial_sums;
    float* d_out_partial_sums;

    cudaMallocHost((void**) &h_out_partial_sums, sizeof(float) * blocks);
    cudaMalloc((void**) &d_out_partial_sums, sizeof(float) * blocks);

    full_sum_kernel<256><<<blocks, num_threads_per_block, num_threads_per_block * sizeof(float)>>>(
        a->data->data, d_out_partial_sums, N);

    cudaMemcpy(h_out_partial_sums, d_out_partial_sums, sizeof(float) * blocks,
               cudaMemcpyDeviceToHost);

    float total_sum = 0.0f;
    for (int i = 0; i < blocks; ++i)
    {
        total_sum += h_out_partial_sums[i];
    }

    out->data = (Storage*) malloc(sizeof(Storage));
    if (!out->data)
    {
        LOG_ERROR("Failed to allocate Storage for out tensor in sum_full_op_cuda");
        SAFE_FREE(&h_out_partial_sums, cudaFreeHost);
        SAFE_FREE(&d_out_partial_sums, cudaFree);
        return;
    }
    out->data->counter = 1;
    out->data->size = 1; // Scalar output

    cudaError_t err = cudaMalloc((void**) &out->data->data, out->data->size * sizeof(float));
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to allocate CUDA memory for out->data->data in sum_full_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data, free);
        SAFE_FREE(&h_out_partial_sums, cudaFreeHost);
        SAFE_FREE(&d_out_partial_sums, cudaFree);
        return;
    }
    err = cudaMemcpy(out->data->data, &total_sum, sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to copy final sum to CUDA device in sum_full_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data->data, cudaFree);
        SAFE_FREE(&out->data, free);
        SAFE_FREE(&h_out_partial_sums, cudaFreeHost);
        SAFE_FREE(&d_out_partial_sums, cudaFree);
        return;
    }

    SAFE_FREE(&h_out_partial_sums, cudaFreeHost);
    SAFE_FREE(&d_out_partial_sums, cudaFree);

    LOG_INFO("Sum operation on cuda done.");
}

void mean_full_op_cuda(Tensor* a, Tensor* out)
{
    LOG_INFO("mean operation on cuda starting......");
    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int blocks = (N + (num_threads_per_block * 2) - 1) / (num_threads_per_block * 2);

    float* h_out_partial_sums;
    float* d_out_partial_sums;

    cudaMallocHost((void**) &h_out_partial_sums, sizeof(float) * blocks);
    cudaMalloc((void**) &d_out_partial_sums, sizeof(float) * blocks);

    full_sum_kernel<256><<<blocks, num_threads_per_block, num_threads_per_block * sizeof(float)>>>(
        a->data->data, d_out_partial_sums, N);

    cudaMemcpy(h_out_partial_sums, d_out_partial_sums, sizeof(float) * blocks,
               cudaMemcpyDeviceToHost);

    float total_sum = 0.0f;
    for (int i = 0; i < blocks; ++i)
    {
        total_sum += h_out_partial_sums[i];
    }

    out->data = (Storage*) malloc(sizeof(Storage));
    if (!out->data)
    {
        LOG_ERROR("Failed to allocate Storage for out tensor in mean_full_op_cuda");
        SAFE_FREE(&h_out_partial_sums, cudaFreeHost);
        SAFE_FREE(&d_out_partial_sums, cudaFree);
        return;
    }
    out->data->counter = 1;
    out->data->size = 1; // Scalar output

    cudaError_t err = cudaMalloc((void**) &out->data->data, out->data->size * sizeof(float));
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to allocate CUDA memory for out->data->data in mean_full_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data, free);
        SAFE_FREE(&h_out_partial_sums, cudaFreeHost);
        SAFE_FREE(&d_out_partial_sums, cudaFree);
        return;
    }
    float final_mean = total_sum / N;
    err = cudaMemcpy(out->data->data, &final_mean, sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to copy final mean to CUDA device in mean_full_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data->data, cudaFree);
        SAFE_FREE(&out->data, free);
        SAFE_FREE(&h_out_partial_sums, cudaFreeHost);
        SAFE_FREE(&d_out_partial_sums, cudaFree);
        return;
    }

    SAFE_FREE(&h_out_partial_sums, cudaFreeHost);
    SAFE_FREE(&d_out_partial_sums, cudaFree);

    LOG_INFO("mean operation on cuda done.");
}

void max_full_op_cuda(Tensor* a, Tensor* out)
{
    LOG_INFO("max operation on cuda starting......");
    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int blocks = (N + (num_threads_per_block * 2) - 1) / (num_threads_per_block * 2);

    float* h_out_partial_maxs;
    float* d_out_partial_maxs;

    cudaMallocHost((void**) &h_out_partial_maxs, sizeof(float) * blocks);
    cudaMalloc((void**) &d_out_partial_maxs, sizeof(float) * blocks);

    full_max_kernel<256><<<blocks, num_threads_per_block, num_threads_per_block * sizeof(float)>>>(
        a->data->data, d_out_partial_maxs, N);

    cudaMemcpy(h_out_partial_maxs, d_out_partial_maxs, sizeof(float) * blocks,
               cudaMemcpyDeviceToHost);

    float max = h_out_partial_maxs[0];
    for (int i = 0; i < blocks; ++i)
    {
        max = fmaxf(max, h_out_partial_maxs[i]);
    }

    out->data = (Storage*) malloc(sizeof(Storage));
    if (!out->data)
    {
        LOG_ERROR("Failed to allocate Storage for out tensor in max_full_op_cuda");
        SAFE_FREE(&h_out_partial_maxs, cudaFreeHost);
        SAFE_FREE(&d_out_partial_maxs, cudaFree);
        return;
    }
    out->data->counter = 1;
    out->data->size = 1; // Scalar output

    cudaError_t err = cudaMalloc((void**) &out->data->data, out->data->size * sizeof(float));
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to allocate CUDA memory for out->data->data in max_full_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data, free);
        SAFE_FREE(&h_out_partial_maxs, cudaFreeHost);
        SAFE_FREE(&d_out_partial_maxs, cudaFree);
        return;
    }
    err = cudaMemcpy(out->data->data, &max, sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to copy final max to CUDA device in max_full_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data->data, cudaFree);
        SAFE_FREE(&out->data, free);
        SAFE_FREE(&h_out_partial_maxs, cudaFreeHost);
        SAFE_FREE(&d_out_partial_maxs, cudaFree);
        return;
    }

    SAFE_FREE(&h_out_partial_maxs, cudaFreeHost);
    SAFE_FREE(&d_out_partial_maxs, cudaFree);

    LOG_INFO("max operation on cuda done.");
}
