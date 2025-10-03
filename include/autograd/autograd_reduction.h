#ifndef AXON_REDUCTION_GRAD
#define AXON_REDUCTION_GRAD

#include "utils.h"
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

#include "autograd/autograd_utils.h"
#include "logger.h"
#include "ops/init_ops.h"

void sum_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);
void mean_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);
void max_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);
void sum_full_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);
void mean_full_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);
void max_full_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);

#endif
