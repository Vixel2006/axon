#ifndef AXON_OPS_CUDA_BINARY_SCALAR_H
#define AXON_OPS_CUDA_BINARY_SCALAR_H

#include "axon_export.h"
#include "logger.h"
#include "ops/binary_scalar_ops.h"
#include "tensor.h"

#define CHECK_CUDA()                                                                               \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = cudaGetLastError();                                                      \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            LOG_ERROR("CUDA runtime error at %s:%d: %s", __FILE__, __LINE__,                       \
                      cudaGetErrorString(err));                                                    \
        }                                                                                          \
    } while (0)

#ifdef __cplusplus
extern "C"
{
#endif

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

#endif // AXON_OPS_CUDA_BINARY_SCALAR_H
