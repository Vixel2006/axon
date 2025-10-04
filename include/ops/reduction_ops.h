#ifndef AXON_REDUCTION_OPS_H
#define AXON_REDUCTION_OPS_H

#include "init_ops.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    AXON_EXPORT void sum_op_cpu(Tensor* a, Tensor* b, int axis, bool keepdim);
    AXON_EXPORT void mean_op_cpu(Tensor* a, Tensor* b, int axis, bool keepdim);
    AXON_EXPORT void max_op_cpu(Tensor* a, Tensor* b, int axis, bool keepdim);
    AXON_EXPORT void sum_full_op_cpu(Tensor* a, Tensor* out);
    AXON_EXPORT void mean_full_op_cpu(Tensor* a, Tensor* out);
    AXON_EXPORT void max_full_op_cpu(Tensor* a, Tensor* out);

    AXON_EXPORT void sum_op_cuda(Tensor* a, Tensor* out, int axis, bool keepdim);
    AXON_EXPORT void mean_op_cuda(Tensor* a, Tensor* out, int axis, bool keepdim);
    AXON_EXPORT void max_op_cuda(Tensor* a, Tensor* out, int axis, bool keepdim);
    AXON_EXPORT void sum_full_op_cuda(Tensor* a, Tensor* out);
    AXON_EXPORT void mean_full_op_cuda(Tensor* a, Tensor* out);
    AXON_EXPORT void max_full_op_cuda(Tensor* a, Tensor* out);

#ifdef __cplusplus
}
#endif

#endif
