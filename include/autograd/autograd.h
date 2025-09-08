#ifndef IDRAK_AUTOGRAD_H
#define IDRAK_AUTOGRAD_H

#include "tensor.h"

void add_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras);
void sub_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras);
void mul_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras);
void div_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras);
void matmul_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras);
void tanh_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras);
void sigmoid_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras);

#endif
