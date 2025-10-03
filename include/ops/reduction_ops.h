#ifndef AXON_REDUCTION_OPS_H
#define AXON_REDUCTION_OPS_H

#include "init_ops.h"
#include "tensor.h"

void sum_op(Tensor* a, Tensor* b, int axis, bool keepdim);
void mean_op(Tensor* a, Tensor* b, int axis, bool keepdim);
void max_op(Tensor* a, Tensor* b, int axis, bool keepdim);
void sum_full_op(Tensor* a, Tensor* out);
void mean_full_op(Tensor* a, Tensor* out);
void max_full_op(Tensor* a, Tensor* out);

#endif
