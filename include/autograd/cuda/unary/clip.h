#ifndef AUTOGRAD_CUDA_UNARY_CLIP_H
#define AUTOGRAD_CUDA_UNARY_CLIP_H

#include "autograd/autograd_unary.h" // For ClipExtras
#include "axon_export.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    AXON_EXPORT void clip_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);

#ifdef __cplusplus
}
#endif

#endif // AUTOGRAD_CUDA_UNARY_CLIP_H
