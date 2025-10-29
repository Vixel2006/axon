#ifndef AUTOGRAD_CUDA_MOVEMENT_COMMON_CUH
#define AUTOGRAD_CUDA_MOVEMENT_COMMON_CUH

#include "autograd/autograd_movement.h"
#include "cuda_utils.h" // For CHECK_CUDA
#include "logger.h"
#include "tensor.h"
#include "utils.h" // For numel, is_contiguous
#include <assert.h>
#include <cuda_runtime.h>
#include <stdlib.h>

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
// Kernel declarations
__global__ void concat_grad_kernel_contiguous(const float* out_grad, float* prev_grad,
                                              size_t outer_size, size_t prev_concat_axis_size,
                                              size_t out_concat_axis_size, size_t inner_size,
                                              size_t offset_in_axis);

__global__ void concat_grad_kernel_noncontiguous(const float* out_grad, float* prev_grad,
                                                 const size_t* prev_strides, int prev_ndim,
                                                 const size_t* prev_shape, int axis,
                                                 size_t outer_size, size_t prev_concat_axis_size,
                                                 size_t out_concat_axis_size, size_t inner_size,
                                                 size_t offset_in_axis);

#endif // AUTOGRAD_CUDA_MOVEMENT_COMMON_CUH
