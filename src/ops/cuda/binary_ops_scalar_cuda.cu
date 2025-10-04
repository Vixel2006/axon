#include "logger.h"
#include "ops/binary_scalar_ops.h"

// TODO: Implement CUDA kernels for binary_ops_scalar
void add_scalar_op_cuda(Tensor* a, float b, Tensor* out)
{
    LOG_WARN("add_scalar_op_cuda: CUDA implementation not available yet.");
}
void sub_scalar_op_cuda(Tensor* a, float b, Tensor* out)
{
    LOG_WARN("sub_scalar_op_cuda: CUDA implementation not available yet.");
}
void rsub_scalar_op_cuda(Tensor* a, float b, Tensor* out)
{
    LOG_WARN("rsub_scalar_op_cuda: CUDA implementation not available yet.");
}
void mul_scalar_op_cuda(Tensor* a, float b, Tensor* out)
{
    LOG_WARN("mul_scalar_op_cuda: CUDA implementation not available yet.");
}
void div_scalar_op_cuda(Tensor* a, float b, Tensor* out)
{
    LOG_WARN("div_scalar_op_cuda: CUDA implementation not available yet.");
}
void rdiv_scalar_op_cuda(Tensor* a, float b, Tensor* out)
{
    LOG_WARN("rdiv_scalar_op_cuda: CUDA implementation not available yet.");
}
void pow_scalar_op_cuda(Tensor* a, float b, Tensor* out)
{
    LOG_WARN("pow_scalar_op_cuda: CUDA implementation not available yet.");
}
