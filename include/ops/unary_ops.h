#ifndef AXON_UNARY_OPS_H
#define AXON_UNARY_OPS_H

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    AXON_EXPORT void relu_op_cpu(Tensor* in, Tensor* out);
    AXON_EXPORT void log_op_cpu(Tensor* in, Tensor* out);
    AXON_EXPORT void exp_op_cpu(Tensor* in, Tensor* out);
    AXON_EXPORT void abs_op_cpu(Tensor* in, Tensor* out);
    AXON_EXPORT void neg_op_cpu(Tensor* in, Tensor* out);
    AXON_EXPORT void clip_op_cpu(Tensor* in, Tensor* out, float min_val, float max_val);

    AXON_EXPORT void relu_op_cuda(Tensor* in, Tensor* out);
    AXON_EXPORT void log_op_cuda(Tensor* in, Tensor* out);
    AXON_EXPORT void exp_op_cuda(Tensor* in, Tensor* out);
    AXON_EXPORT void abs_op_cuda(Tensor* in, Tensor* out);
    AXON_EXPORT void neg_op_cuda(Tensor* in, Tensor* out);
    AXON_EXPORT void clip_op_cuda(Tensor* in, Tensor* out, float min_val, float max_val);

#ifdef __cplusplus
}
#endif

#endif
