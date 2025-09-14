#ifndef IDRAK_BINARY_SCALAR_H
#define IDRAK_BINARY_SCALAR_H

#include "ops/init_ops.h"
#include "tensor.h"

void add_scalar_op(Tensor* a, float b, Tensor* out);
void sub_scalar_op(Tensor* a, float b, Tensor* out);
void rsub_scalar_op(Tensor* a, float b, Tensor* out);
void mul_scalar_op(Tensor* a, float b, Tensor* out);
void div_scalar_op(Tensor* a, float b, Tensor* out);
void rdiv_scalar_op(Tensor* a, float b, Tensor* out);
void pow_scalar_op(Tensor* a, float b, Tensor* out);

#endif
