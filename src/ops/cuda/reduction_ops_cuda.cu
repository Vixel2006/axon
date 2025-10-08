#include "logger.h"
#include "ops/reduction_ops.h"
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void sum_kernel(float* a, float* out, int n, int offset, int axis) {}
__global__ void mean_kernel() {}
__global__ void max_kernel() {}

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
    LOG_WARN("sum_op_cuda: CUDA implementation not available yet.");
}

void mean_op_cuda(Tensor* a, Tensor* out, int axis, bool keepdim)
{
    LOG_WARN("mean_op_cuda: CUDA implementation not available yet.");
}

void max_op_cuda(Tensor* a, Tensor* out, int axis, bool keepdim)
{
    LOG_WARN("max_op_cuda: CUDA implementation not available yet.");
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

    float* final_sum_ptr = (float*) malloc(sizeof(float));
    if (!final_sum_ptr)
    {
        LOG_ERROR("sum_full_op_cuda: Failed to allocate memory for final_sum_ptr");
        SAFE_FREE(&h_out_partial_sums, cudaFreeHost);
        SAFE_FREE(&d_out_partial_sums, cudaFree);
        return;
    }
    final_sum_ptr[0] = total_sum;

    from_data(out, final_sum_ptr);
    SAFE_FREE(&final_sum_ptr, free);

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

    float* final_sum_ptr = (float*) malloc(sizeof(float));
    if (!final_sum_ptr)
    {
        LOG_ERROR("mean_full_op_cuda: Failed to allocate memory for final_mean_ptr");
        SAFE_FREE(&h_out_partial_sums, cudaFreeHost);
        SAFE_FREE(&d_out_partial_sums, cudaFree);
        return;
    }
    final_sum_ptr[0] = total_sum / N;

    from_data(out, final_sum_ptr);
    SAFE_FREE(&final_sum_ptr, free);

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

    float* final_max_ptr = (float*) malloc(sizeof(float));
    if (!final_max_ptr)
    {
        LOG_ERROR("max_full_op_cuda: Failed to allocate memory for final_max_ptr");
        SAFE_FREE(&h_out_partial_maxs, cudaFreeHost);
        SAFE_FREE(&d_out_partial_maxs, cudaFree);
        return;
    }
    final_max_ptr[0] = max;

    from_data(out, final_max_ptr);
    SAFE_FREE(&final_max_ptr, free);

    SAFE_FREE(&h_out_partial_maxs, cudaFreeHost);
    SAFE_FREE(&d_out_partial_maxs, cudaFree);

    LOG_INFO("max operation on cuda done.");
}
