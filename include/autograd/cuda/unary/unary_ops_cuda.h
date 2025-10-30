#ifndef AXON_AUTOGRAD_CUDA_UNARY_OPS_CUDA_H
#define AXON_AUTOGRAD_CUDA_UNARY_OPS_CUDA_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

void log_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
void exp_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
void abs_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
void neg_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
void relu_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
void clip_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);

#ifdef __cplusplus
}
#endif

#endif // AXON_AUTOGRAD_CUDA_UNARY_OPS_CUDA_H
