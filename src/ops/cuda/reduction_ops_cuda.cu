#include "logger.h"
#include "ops/reduction_ops.h"

// TODO: Implement CUDA kernels for reduction_ops
void sum_op_cuda(Tensor* a, Tensor* out, int axis, bool keepdim)
{
    LOG_WARN("sum_op_cuda: CUDA implementation not available yet.");
}
void mean_op_cuda(Tensor* a, Tensor* out, int axis, bool keepdim)
{
    LOG_WARN("mean_op_cuda: CUDA implementation not available yet.");
}
void max_op_cuda(Tensor* a, Tensor* out, int axis, bool keepdim)
{
    LOG_WARN("max_op_cuda: CUDA implementation not available yet.");
}
void sum_full_op_cuda(Tensor* a, Tensor* out)
{
    LOG_WARN("sum_full_op_cuda: CUDA implementation not available yet.");
}
void mean_full_op_cuda(Tensor* a, Tensor* out)
{
    LOG_WARN("mean_full_op_cuda: CUDA implementation not available yet.");
}
void max_full_op_cuda(Tensor* a, Tensor* out)
{
    LOG_WARN("max_full_op_cuda: CUDA implementation not available yet.");
}
