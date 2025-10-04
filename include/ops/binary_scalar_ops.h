#ifndef AXON_BINARY_SCALAR_H
#define AXON_BINARY_SCALAR_H

#include "ops/init_ops.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    AXON_EXPORT void add_scalar_op_cpu(Tensor* a, float b, Tensor* out);
    AXON_EXPORT void sub_scalar_op_cpu(Tensor* a, float b, Tensor* out);
    AXON_EXPORT void rsub_scalar_op_cpu(Tensor* a, float b, Tensor* out);
    AXON_EXPORT void mul_scalar_op_cpu(Tensor* a, float b, Tensor* out);
    AXON_EXPORT void div_scalar_op_cpu(Tensor* a, float b, Tensor* out);
    AXON_EXPORT void rdiv_scalar_op_cpu(Tensor* a, float b, Tensor* out);
    AXON_EXPORT void pow_scalar_op_cpu(Tensor* a, float b, Tensor* out);

    AXON_EXPORT void add_scalar_op_cuda(Tensor* a, float b, Tensor* out);
    AXON_EXPORT void sub_scalar_op_cuda(Tensor* a, float b, Tensor* out);
    AXON_EXPORT void rsub_scalar_op_cuda(Tensor* a, float b, Tensor* out);
    AXON_EXPORT void mul_scalar_op_cuda(Tensor* a, float b, Tensor* out);
    AXON_EXPORT void div_scalar_op_cuda(Tensor* a, float b, Tensor* out);
    AXON_EXPORT void rdiv_scalar_op_cuda(Tensor* a, float b, Tensor* out);
    AXON_EXPORT void pow_scalar_op_cuda(Tensor* a, float b, Tensor* out);

#ifdef __cplusplus
}
#endif

#endif
