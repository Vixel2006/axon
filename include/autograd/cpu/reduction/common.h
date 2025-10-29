#ifndef AUTOGRAD_CPU_REDUCTION_COMMON_H
#define AUTOGRAD_CPU_REDUCTION_COMMON_H

#include "autograd/autograd_reduction.h"
#include "autograd/autograd_utils.h" // For get_reduced_dim
#include "logger.h"
#include "tensor.h"
#include "utils.h" // For numel, is_contiguous
#include <immintrin.h> // For SIMD operations
#include <stdlib.h>    // For malloc, free, exit

#define SIMD_WIDTH 8
#define MAX_DIMS 32

// Helper function to properly map input indices to output indices
static void map_in_coords_to_out_offset(int* in_coords, int in_ndim, int reduced_dim, Tensor* out,
                                        int* out_offset_result)
{
    int out_idx = 0;
    for (int d = 0; d < in_ndim; ++d)
    {
        if (d != reduced_dim)
        {
            // This dimension exists in output
            int out_coord = in_coords[d];
            // Find which output dimension this maps to
            int out_d = d;
            if (d > reduced_dim && out->ndim < in_ndim)
            {
                out_d = d - 1; // Adjust for removed dimension
            }
            if (out_d < out->ndim)
            {
                out_idx += out_coord * out->strides[out_d];
            }
        }
    }
    *out_offset_result = out_idx;
}

#endif // AUTOGRAD_CPU_REDUCTION_COMMON_H
