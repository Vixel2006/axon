#ifndef IDRAK_OPS_UTILS_H
#define IDRAK_OPS_UTILS_H

#include <string.h>

#include "tensor.h"
#include "shared_ptr.h"

int tensor_alloc_shape(int ndim, const int *shape, int **out_shape);
int tensor_alloc_strides(int ndim, const int *strides, int **out_strides);
int tensor_copy_layout(Tensor *in, Tensor *out, const int *shape);
void reference_shared_ptr(shared_ptr *out, shared_ptr in);
void tensor_init_view(Tensor *out, Tensor *in);

#endif
