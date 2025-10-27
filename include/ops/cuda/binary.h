#ifndef AXON_OPS_CUDA_BINARY_H
#define AXON_OPS_CUDA_BINARY_H

#include "logger.h"
#include "ops/binary_ops.h"
#include "ops/init_ops.h"
#include "utils/indexing.cuh"
#include <assert.h>
#include <cuda_runtime.h>

#define TILE_DIM 16

#define CHECK_CUDA()                                                                               \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = cudaGetLastError();                                                      \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            LOG_ERROR("CUDA runtime error at %s:%d: %s", __FILE__, __LINE__,                       \
                      cudaGetErrorString(err));                                                    \
            assert(0 && "CUDA runtime error");                                                     \
        }                                                                                          \
    } while (0)

__global__ void add_kernel(const float* a, const float* b, float* out, const int n);
__global__ void sub_kernel(const float* a, const float* b, float* out, const int n);
__global__ void mul_kernel(const float* a, const float* b, float* out, const int n);
__global__ void div_kernel(const float* a, const float* b, float* out, const int n);
__global__ void pow_kernel(const float* a, const float* b, float* out, const int n);
__global__ void matmul_kernel(const float* a, const float* b, float* out, const int N, const int M,
                              const int K);

#endif // AXON_OPS_CUDA_BINARY_H
