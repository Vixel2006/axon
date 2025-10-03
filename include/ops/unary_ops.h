#ifndef AXON_UNARY_OPS_H
#define AXON_UNARY_OPS_H

#include "tensor.h"

void relu_op(Tensor* in, Tensor* out);
void log_op(Tensor* in, Tensor* out);
void exp_op(Tensor* in, Tensor* out);
void abs_op(Tensor* in, Tensor* out);
void neg_op(Tensor* in, Tensor* out);

#endif
