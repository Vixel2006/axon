#ifndef AXON_AUTOGRAD_UTILS_H
#define AXON_AUTOGRAD_UTILS_H

#include "tensor.h"

// Struct to pass min_val and max_val to clip_grad_op
typedef struct
{
    double min_val;
    double max_val;
} ClipExtras;

int get_reduced_dim(int* in_shape, int* out_shape, int in_ndim, int out_ndim);

int get_num_reduction_batches(int* in_shape, int in_ndim, int reduced_dim);

#endif
