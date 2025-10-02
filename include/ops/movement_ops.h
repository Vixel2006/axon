#ifndef IDRAK_MOVEMENT_OPS_H
#define IDRAK_MOVEMENT_OPS_H

#include "init_ops.h"
#include "tensor.h"

void view_op(Tensor* in, Tensor* out, int* shape, int ndim);
void unsqueeze_op(Tensor* in, Tensor* out, int dim);
void squeeze_op(Tensor* in, Tensor* out, int dim);
void transpose_op(Tensor* in, Tensor* out, int N, int M);
void expand_op(Tensor* in, Tensor* out, const int* shape);
void broadcast_op(Tensor* in, Tensor* out, int ndim, const int* shape);
void concat_op(Tensor** in, Tensor* out, int num_tensors, int axis);

#endif
