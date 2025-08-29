#ifndef NAWAH_AUTOGRAD_UTILS_H
#define NAWAH_AUTOGRAD_UTILS_H

#include "tensor.h"

int get_reduced_dim(int *in_shape, int *out_shape, int in_ndim, int out_ndim);

int get_num_reduction_batches(int *in_shape, int in_ndim, int reduced_dim);

#endif