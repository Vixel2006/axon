#ifndef AUTOGRAD_CPU_BINARY_CONV2D_H
#define AUTOGRAD_CPU_BINARY_CONV2D_H

#include "axon_export.h"
#include "tensor.h"
#include "autograd/autograd_binary.h" // For BackwardConvExtras

#ifdef __cplusplus
extern "C"
{
#endif

    AXON_EXPORT void conv2d_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);

#ifdef __cplusplus
}
#endif

#endif // AUTOGRAD_CPU_BINARY_CONV2D_H
