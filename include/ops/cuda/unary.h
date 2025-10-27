#ifndef AXON_OPS_CUDA_UNARY_H
#define AXON_OPS_CUDA_UNARY_H

#include "logger.h"
#include "ops/init_ops.h"
#include "ops/unary_ops.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

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

__global__ void relu_kernel(const float* a, float* b, int n);
__global__ void log_kernel(const float* a, float* b, int n);
__global__ void exp_kernel(const float* a, float* b, int n);
__global__ void neg_kernel(const float* a, float* b, int n);
__global__ void abs_kernel(const float* a, float* b, int n);
__global__ void clip_kernel(const float* a, float* b, const float min_val, const float max_val,
                            int n);

#endif // AXON_OPS_CUDA_UNARY_H
