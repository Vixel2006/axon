#ifndef AUTOGRAD_CPU_BINARY_MATMUL_H
#define AUTOGRAD_CPU_BINARY_MATMUL_H

#include "axon_export.h"
#include "tensor.h"
#include "autograd/autograd_binary.h" // For MatMulBackwardExtras

#ifdef __cplusplus
extern "C"
{
#endif

    AXON_EXPORT void matmul_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);

#ifdef __cplusplus
}
#endif

#endif // AUTOGRAD_CPU_BINARY_MATMUL_H
