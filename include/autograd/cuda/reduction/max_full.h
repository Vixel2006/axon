#ifndef AUTOGRAD_CUDA_REDUCTION_MAX_FULL_H
#define AUTOGRAD_CUDA_REDUCTION_MAX_FULL_H

#include "axon_export.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    AXON_EXPORT void max_full_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);

#ifdef __cplusplus
}
#endif

#endif // AUTOGRAD_CUDA_REDUCTION_MAX_FULL_H
