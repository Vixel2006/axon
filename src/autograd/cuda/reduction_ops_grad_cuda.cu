#include "autograd/autograd_reduction.h"
#include "cuda_utils.h"
#include "logger.h"

// CUDA kernel for mean_full_grad_op_cuda
__global__ void mean_full_grad_kernel(float* in_grad_data, float output_grad, int in_size,
                                      float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < in_size)
    {
        in_grad_data[idx] += output_grad * scale;
    }
}

void sum_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_WARN("sum_grad_op_cuda: CUDA implementation not available yet.");
}
void mean_full_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("GRAD: mean_full_grad_op_cuda: Computing gradient for full mean reduction (CUDA)");
}

void mean_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("GRAD: mean_grad_op_cuda: Computing gradient for mean reduction (CUDA)");
}

void max_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_WARN("max_grad_op_cuda: CUDA implementation not available yet.");
}
void sum_full_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_WARN("sum_full_grad_op_cuda: CUDA implementation not available yet.");
}
void max_full_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_WARN("max_full_grad_op_cuda: CUDA implementation not available yet.");
}
