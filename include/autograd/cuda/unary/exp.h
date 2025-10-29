#ifndef AUTOGRAD_CUDA_UNARY_EXP_H
#define AUTOGRAD_CUDA_UNARY_EXP_H

#include "axon_export.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    AXON_EXPORT void exp_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);

#ifdef __cplusplus
}
#endif

#endif // AUTOGRAD_CUDA_UNARY_EXP_H
