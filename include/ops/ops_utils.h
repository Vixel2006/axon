#ifndef NAWAH_OPS_UTILS_H
#define NAWAH_OPS_UTILS_H

#include <string.h>

#include "tensor.h"

int tensor_alloc_shape(int ndim, const int *shape, int **out_shape);
int tensor_alloc_strides(int ndim, const int *strides, int **out_strides);
int tensor_copy_layout(Tensor *in, Tensor *out, const int *shape);
void tensor_init_view(Tensor *out, Tensor *in);

#endif