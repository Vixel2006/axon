#include "logger.h"
#include "ops/movement_ops.h"

// TODO: Implement CUDA kernels for movement_ops
void view_op_cuda(Tensor* in, Tensor* out, int* shape, int ndim)
{
    LOG_WARN("view_op_cuda: CUDA implementation not available yet.");
}
void unsqueeze_op_cuda(Tensor* in, Tensor* out, int dim)
{
    LOG_WARN("unsqueeze_op_cuda: CUDA implementation not available yet.");
}
void squeeze_op_cuda(Tensor* in, Tensor* out, int dim)
{
    LOG_WARN("squeeze_op_cuda: CUDA implementation not available yet.");
}
void transpose_op_cuda(Tensor* in, Tensor* out, int N, int M)
{
    LOG_WARN("transpose_op_cuda: CUDA implementation not available yet.");
}
void expand_op_cuda(Tensor* in, Tensor* out, const int* shape)
{
    LOG_WARN("expand_op_cuda: CUDA implementation not available yet.");
}
void broadcast_op_cuda(Tensor* in, Tensor* out, int ndim, const int* shape)
{
    LOG_WARN("broadcast_op_cuda: CUDA implementation not available yet.");
}
void concat_op_cuda(Tensor** in, Tensor* out, int num_tensors, int axis)
{
    LOG_WARN("concat_op_cuda: CUDA implementation not available yet.");
}
