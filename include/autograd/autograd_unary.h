#ifndef AXON_UNARY_GRAD
#define AXON_UNARY_GRAD
#include "logger.h"
#include "utils.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>
#include <string.h>

typedef struct
{
    float min_val;
    float max_val;
} ClipExtras;

void relu_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);
void abs_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);
void log_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);
void exp_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);
void neg_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);
void clip_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);

#endif
