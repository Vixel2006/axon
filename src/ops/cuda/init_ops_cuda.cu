#include "logger.h"
#include "ops/init_ops.h"

// TODO: Implement CUDA kernels for init_ops
void zeros_cuda(Tensor* t) { LOG_WARN("zeros_cuda: CUDA implementation not available yet."); }
void ones_cuda(Tensor* t) { LOG_WARN("ones_cuda: CUDA implementation not available yet."); }
void randn_cuda(Tensor* t) { LOG_WARN("randn_cuda: CUDA implementation not available yet."); }
void uniform_cuda(Tensor* t, float low, float high)
{
    LOG_WARN("uniform_cuda: CUDA implementation not available yet.");
}
void from_data_cuda(Tensor* t, float* data)
{
    LOG_WARN("from_data_cuda: CUDA implementation not available yet.");
}
