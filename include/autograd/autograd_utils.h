#ifndef AXON_AUTOGRAD_UTILS_H
#define AXON_AUTOGRAD_UTILS_H

#include "tensor.h"

#include "axon_export.h" // Include the generated export header

#ifdef __cplusplus
extern "C"
{
#endif
    // Struct to pass min_val and max_val to clip_grad_op
    typedef struct
    {
        double min_val;
        double max_val;
    } ClipExtras;

    AXON_EXPORT int get_reduced_dim(int* in_shape, int* out_shape, int in_ndim, int out_ndim);
    AXON_EXPORT int get_num_reduction_batches(int* in_shape, int in_ndim, int reduced_dim);
#ifdef __cplusplus
}
#endif

#endif
