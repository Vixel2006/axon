#include "ops/cuda/binary_scalar.h"

void div_scalar_op_cuda(Tensor* a, float b, Tensor* out)
{
    LOG_DEBUG("div_scalar_op_cuda: Entering function with scalar=%.2f", b);
    LOG_WARN("div_scalar_op_cuda: CUDA implementation not available yet.");
}
