#ifndef AXON_MOVEMENT_OPS_H
#define AXON_MOVEMENT_OPS_H

#include "init_ops.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    AXON_EXPORT void view_op(Tensor* in, Tensor* out, int* shape, int ndim);
    AXON_EXPORT void unsqueeze_op(Tensor* in, Tensor* out, int dim);
    AXON_EXPORT void squeeze_op(Tensor* in, Tensor* out, int dim);
    AXON_EXPORT void transpose_op(Tensor* in, Tensor* out, int N, int M);
    AXON_EXPORT void expand_op(Tensor* in, Tensor* out, const int* shape);
    AXON_EXPORT void broadcast_op(Tensor* in, Tensor* out, int ndim, const int* shape);

    AXON_EXPORT void concat_op_cpu(Tensor** in, Tensor* out, int num_tensors, int axis);
    AXON_EXPORT void concat_op_cuda(Tensor** in, Tensor* out, int num_tensors, int axis);

#ifdef __cplusplus
}
#endif

#endif
