#ifndef AXON_REDUCTION_GRAD
#define AXON_REDUCTION_GRAD

#include "utils.h"
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

#include "autograd/autograd_utils.h"
#include "axon_export.h" // Include the generated export header
#include "logger.h"
#include "ops/init_ops.h"

#ifdef __cplusplus
extern "C"
{
#endif
    AXON_EXPORT void sum_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void mean_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void max_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void sum_full_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void mean_full_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void max_full_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);

    AXON_EXPORT void sum_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void mean_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void max_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void sum_full_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void mean_full_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void max_full_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
#ifdef __cplusplus
}
#endif

#endif
