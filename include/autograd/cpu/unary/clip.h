#ifndef AUTOGRAD_CPU_UNARY_CLIP_H
#define AUTOGRAD_CPU_UNARY_CLIP_H

#include "axon_export.h"
#include "tensor.h"
#include "autograd/autograd_unary.h" // For ClipExtras

#ifdef __cplusplus
extern "C"
{
#endif

    AXON_EXPORT void clip_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);

#ifdef __cplusplus
}
#endif

#endif // AUTOGRAD_CPU_UNARY_CLIP_H
