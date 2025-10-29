#ifndef AUTOGRAD_CPU_UNARY_ABS_H
#define AUTOGRAD_CPU_UNARY_ABS_H

#include "axon_export.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    AXON_EXPORT void abs_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);

#ifdef __cplusplus
}
#endif

#endif // AUTOGRAD_CPU_UNARY_ABS_H
