#ifndef AXON_BINARY_OPS_H
#define AXON_BINARY_OPS_H

#include "tensor.h"

#include "axon_export.h" // Include the generated export header

#ifdef __cplusplus
extern "C"
{
#endif

    AXON_EXPORT void add_op_cpu(Tensor* a, Tensor* b, Tensor* out);
    AXON_EXPORT void sub_op_cpu(Tensor* a, Tensor* b, Tensor* out);
    AXON_EXPORT void mul_op_cpu(Tensor* a, Tensor* b, Tensor* out);
    AXON_EXPORT void div_op_cpu(Tensor* a, Tensor* b, Tensor* out);
    AXON_EXPORT void pow_op_cpu(Tensor* a, Tensor* b, Tensor* out);
    AXON_EXPORT void matmul_op_cpu(Tensor* a, Tensor* b, Tensor* out, int N, int K, int P);
    AXON_EXPORT void dot_op_cpu(Tensor* a, Tensor* b, Tensor* out);
    AXON_EXPORT void conv2d_op_cpu(Tensor* in, Tensor* kernel, Tensor* out, const int* kernel_size,
                                   const int* stride, int padding);

    AXON_EXPORT void add_op_cuda(Tensor* a, Tensor* b, Tensor* out);
    AXON_EXPORT void sub_op_cuda(Tensor* a, Tensor* b, Tensor* out);
    AXON_EXPORT void mul_op_cuda(Tensor* a, Tensor* b, Tensor* out);
    AXON_EXPORT void div_op_cuda(Tensor* a, Tensor* b, Tensor* out);
    AXON_EXPORT void pow_op_cuda(Tensor* a, Tensor* b, Tensor* out);
    AXON_EXPORT void matmul_op_cuda(Tensor* a, Tensor* b, Tensor* out, int N, int K, int P);
    AXON_EXPORT void dot_op_cuda(Tensor* a, Tensor* b, Tensor* out);
    AXON_EXPORT void conv2d_op_cuda(Tensor* in, Tensor* kernel, Tensor* out, const int* kernel_size,
                                    const int* stride, int padding);

#ifdef __cplusplus
}
#endif

#endif
