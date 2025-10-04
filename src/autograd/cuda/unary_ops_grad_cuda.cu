#include "autograd/autograd_unary.h"
#include "logger.h"

// TODO: Implement CUDA kernels for unary_ops_grad
void relu_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_WARN("relu_grad_op_cuda: CUDA implementation not available yet.");
}
void log_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_WARN("log_grad_op_cuda: CUDA implementation not available yet.");
}
void exp_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_WARN("exp_grad_op_cuda: CUDA implementation not available yet.");
}
void abs_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_WARN("abs_grad_op_cuda: CUDA implementation not available yet.");
}
void neg_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_WARN("neg_grad_op_cuda: CUDA implementation not available yet.");
}
void clip_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_WARN("clip_grad_op_cuda: CUDA implementation not available yet.");
}
