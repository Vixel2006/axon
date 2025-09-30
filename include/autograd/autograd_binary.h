#ifndef IDRAK_BINARY_GRAD
#define IDRAK_BINARY_GRAD
#include "logger.h"
#include "utils.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>
#include <string.h>

typedef struct {
    int padding;
    int H_in;
    int W_in;
    int Kh;
    int Kw;
    int Sh;
    int Sw;
    int Hout;
    int Wout;
} BackwardConvExtras;

void add_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);
void sub_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);
void rsub_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);
void mul_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);
void div_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);
void rdiv_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);
void matmul_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);
void conv2d_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);
void dot_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);
void pow_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);

#endif
