#ifndef AUTOGRAD_CUDA_REDUCTION_COMMON_CUH
#define AUTOGRAD_CUDA_REDUCTION_COMMON_CUH

#include "autograd/autograd_reduction.h"
#include "cuda_utils.h" // For CHECK_CUDA
#include "logger.h"
#include "tensor.h"
#include "utils.h" // For numel
#include <assert.h>
#include <cuda_runtime.h>

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

typedef struct
{
    int axis;
} ReductionExtras;

// Kernel declarations
__global__ void sum_full_grad_kernel(float* in_grad_data, float* output_grad, int in_size,
                                     const int* in_grad_shape, const int* in_grad_strides,
                                     int in_grad_ndim);
__global__ void mean_full_grad_kernel(float* in_grad_data, float* output_grad, int in_size,
                                     const int* in_grad_shape, const int* in_grad_strides,
                                     int in_grad_ndim);
__global__ void max_full_grad_kernel(float* in_grad_data, float* in_data, float* output_grad,
                                     int in_size, float* max, const int* in_grad_shape,
                                     const int* in_grad_strides, int in_grad_ndim);
__global__ void sum_grad_kernel(const float* out_grad, float* in_grad, const int* shape, int ndim,
                                int axis, int n, const int* in_grad_shape,
                                const int* in_grad_strides, int in_grad_ndim);
__global__ void mean_grad_kernel(const float* out_grad, float* in_grad, const int* shape, int ndim,
                                 int axis, int n, const int* in_grad_shape,
                                 const int* in_grad_strides, int in_grad_ndim);
__global__ void max_grad_kernel(const float* out_grad, float* in_grad, const float* in_data,
                                const float* out_data, const int* shape, const int* in_strides,
                                const int* out_strides, int in_ndim, int out_ndim, int reduced_dim,
                                int n, const int* in_grad_shape, const int* in_grad_strides,
                                int in_grad_ndim);

#endif // AUTOGRAD_CUDA_REDUCTION_COMMON_CUH
