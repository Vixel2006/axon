#ifndef IDRAK_BINARY_OPS_H
#define IDRAK_BINARY_OPS_H

#include "tensor.h"

void add_op(Tensor* a, Tensor* b, Tensor* out);
void sub_op(Tensor* a, Tensor* b, Tensor* out);
void mul_op(Tensor* a, Tensor* b, Tensor* out);
void div_op(Tensor* a, Tensor* b, Tensor* out);
void pow_op(Tensor* a, Tensor* b, Tensor* out);
void matmul_op(Tensor* a, Tensor* b, Tensor* out, int N, int K, int P);
void dot_op(Tensor* a, Tensor* b, Tensor* out);
void conv2d_op(Tensor* in, Tensor* kernel, Tensor* out, const int* kernel_size, const int* stride, int padding);

#endif
