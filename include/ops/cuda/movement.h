#ifndef AXON_OPS_CUDA_MOVEMENT_H
#define AXON_OPS_CUDA_MOVEMENT_H

#include "logger.h"
#include "ops/movement_ops.h"
#include <assert.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(err)                                                                            \
    do                                                                                             \
    {                                                                                              \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            LOG_ERROR("CUDA runtime error at %s:%d: %s", __FILE__, __LINE__,                       \
                      cudaGetErrorString(err));                                                    \
            assert(0 && "CUDA runtime error");                                                     \
        }                                                                                          \
    } while (0)

__global__ void contig_concat_kernel(const float* in_data, float* out_data, size_t outer_size,
                                     size_t in_concat_axis_size, size_t out_concat_axis_size,
                                     size_t inner_size, size_t offset_in_axis);

__global__ void uncontig_concat_kernel(const float* in_data, float* out_data,
                                       const size_t* in_strides, int in_ndim,
                                       const size_t* in_shape, int axis, size_t outer_size,
                                       size_t in_concat_axis_size, size_t out_concat_axis_size,
                                       size_t inner_size, size_t offset_in_axis);

#endif // AXON_OPS_CUDA_MOVEMENT_H