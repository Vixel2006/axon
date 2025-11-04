#ifndef AXON_OPS_CUDA_REDUCTION_H
#define AXON_OPS_CUDA_REDUCTION_H

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
    if (block_size >= 64)
    {
        if (tid < 32) rdata[tid] += rdata[tid + 32];
        __syncthreads();
    }
    if (block_size >= 32)
    {
        if (tid < 16) rdata[tid] += rdata[tid + 16];
        __syncthreads();
    }
    if (block_size >= 16)
    {
        if (tid < 8) rdata[tid] += rdata[tid + 8];
        __syncthreads();
    }
    if (block_size >= 8)
    {
        if (tid < 4) rdata[tid] += rdata[tid + 4];
        __syncthreads();
    }
    if (block_size >= 4)
    {
        if (tid < 2) rdata[tid] += rdata[tid + 2];
        __syncthreads();
    }
    if (block_size >= 2)
    {
        if (tid < 1) rdata[tid] += rdata[tid + 1];
        __syncthreads();
    }
}

template <int block_size>
__global__ void sum_kernel(float* a, float* out, int n, int axis_dim, int outer_dim, int inner_dim);

template <int block_size>
__global__ void mean_kernel(float* a, float* out, int n, int axis_dim, int outer_dim,
                            int inner_dim);

template <int block_size>
__device__ __forceinline__ void wrap_max_reduce(volatile float* rdata, int tid)
{
    if (block_size >= 64)
    {
        if (tid < 32) rdata[tid] = fmaxf(rdata[tid], rdata[tid + 32]);
        __syncthreads();
    }
    if (block_size >= 32)
    {
        if (tid < 16) rdata[tid] = fmaxf(rdata[tid], rdata[tid + 16]);
        __syncthreads();
    }
    if (block_size >= 16)
    {
        if (tid < 8) rdata[tid] = fmaxf(rdata[tid], rdata[tid + 8]);
        __syncthreads();
    }
    if (block_size >= 8)
    {
        if (tid < 4) rdata[tid] = fmaxf(rdata[tid], rdata[tid + 4]);
        __syncthreads();
    }
    if (block_size >= 4)
    {
        if (tid < 2) rdata[tid] = fmaxf(rdata[tid], rdata[tid + 2]);
        __syncthreads();
    }
    if (block_size >= 2)
    {
        if (tid < 1) rdata[tid] = fmaxf(rdata[tid], rdata[tid + 1]);
        __syncthreads();
    }
}

template <int block_size>
__global__ void max_kernel(float* a, float* out, int n, int axis_dim, int inner_dim, int outer_dim);

template <int block_size> __global__ void full_sum_kernel(const float* a, float* out, int n);

template <int block_size> __global__ void full_max_kernel(float* a, float* out, int n);

#endif // AXON_OPS_CUDA_REDUCTION_H
