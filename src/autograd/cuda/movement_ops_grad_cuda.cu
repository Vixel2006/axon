#include "autograd/autograd_movement.h"
#include "logger.h"

// TODO: Implement CUDA kernels for movement_ops_grad
void concat_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_WARN("concat_grad_op_cuda: CUDA implementation not available yet.");
}
