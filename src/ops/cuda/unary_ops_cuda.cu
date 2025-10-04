#include "logger.h"
#include "ops/unary_ops.h"
#include <cuda_runtime.h>

void relu_op_cuda(Tensor* in, Tensor* out)
{
    (void) in;
    (void) out;
    LOG_WARN("relu_op_cuda: CUDA implementation not available yet.");
}

void log_op_cuda(Tensor* in, Tensor* out)
{
    (void) in;
    (void) out;
    LOG_WARN("log_op_cuda: CUDA implementation not available yet.");
}

void exp_op_cuda(Tensor* in, Tensor* out)
{
    (void) in;
    (void) out;
    LOG_WARN("exp_op_cuda: CUDA implementation not available yet.");
}

void neg_op_cuda(Tensor* in, Tensor* out)
{
    (void) in;
    (void) out;
    LOG_WARN("neg_op_cuda: CUDA implementation not available yet.");
}

void clip_op_cuda(Tensor* in, Tensor* out, float min_val, float max_val)
{
    (void) in;
    (void) out;
    (void) min_val;
    (void) max_val;
    LOG_WARN("clip_op_cuda: CUDA implementation not available yet.");
}

void abs_op_cuda(Tensor* in, Tensor* out)
{
    (void) in;
    (void) out;
    LOG_WARN("abs_op_cuda: CUDA implementation not available yet.");
}

void tanh_op_cuda(Tensor* in, Tensor* out)
{
    (void) in;
    (void) out;
    LOG_WARN("tanh_op_cuda: CUDA implementation not available yet.");
}

void sigmoid_op_cuda(Tensor* in, Tensor* out)
{
    (void) in;
    (void) out;
    LOG_WARN("sigmoid_op_cuda: CUDA implementation not available yet.");
}