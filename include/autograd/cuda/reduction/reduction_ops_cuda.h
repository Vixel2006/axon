#ifndef AXON_AUTOGRAD_CUDA_REDUCTION_OPS_CUDA_H
#define AXON_AUTOGRAD_CUDA_REDUCTION_OPS_CUDA_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

void sum_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
void mean_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
void max_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
void sum_full_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
void mean_full_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
void max_full_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);

#ifdef __cplusplus
}
#endif

#endif // AXON_AUTOGRAD_CUDA_REDUCTION_OPS_CUDA_H
