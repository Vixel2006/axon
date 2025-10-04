#include "autograd/autograd_binary.h"
#include "logger.h"
#include "tensor.h"       // Required for Tensor structure
#include <cuda_runtime.h> // Required for CUDA functions

// CUDA kernel for adding gradients
__global__ void add_grad_kernel(const float* out_grad, float* prev_grad, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += out_grad[i];
    }
}

void add_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("add_grad_op_cuda: CUDA implementation called.");

    // For addition, the gradient flows directly to both inputs.
    // So, we add 'out->grad->data' to 'prev[0]->grad->data' and 'prev[1]->grad->data'.
    // Assuming 'prev' contains two tensors for binary operation.

    if (n_prev != 2)
    {
        LOG_ERROR("add_grad_op_cuda expects 2 previous tensors for binary addition.");
        return;
    }

    int N = numel(out->shape, out->ndim); // Number of elements in the output tensor

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    // Accumulate gradient for the first previous tensor
    if (prev[0]->requires_grad)
    {
        add_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data, prev[0]->grad->data,
                                                               N);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            LOG_ERROR("add_grad_op_cuda: Kernel launch failed for prev[0]: %s",
                      cudaGetErrorString(err));
            return;
        }
    }

    // Accumulate gradient for the second previous tensor
    if (prev[1]->requires_grad)
    {
        add_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data, prev[1]->grad->data,
                                                               N);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            LOG_ERROR("add_grad_op_cuda: Kernel launch failed for prev[1]: %s",
                      cudaGetErrorString(err));
            return;
        }
    }
    LOG_INFO("add_grad_op_cuda: CUDA implementation finished successfully.");
}

// TODO: Implement CUDA kernels for other binary_ops_grad
void sub_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_WARN("sub_grad_op_cuda: CUDA implementation not available yet.");
}
void rsub_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_WARN("rsub_grad_op_cuda: CUDA implementation not available yet.");
}
void mul_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_WARN("mul_grad_op_cuda: CUDA implementation not available yet.");
}
void pow_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_WARN("pow_grad_op_cuda: CUDA implementation not available yet.");
}
void div_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_WARN("div_grad_op_cuda: CUDA implementation not available yet.");
}
void rdiv_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_WARN("rdiv_grad_op_cuda: CUDA implementation not available yet.");
}
void matmul_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_WARN("matmul_grad_op_cuda: CUDA implementation not available yet.");
}
void conv2d_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_WARN("conv2d_grad_op_cuda: CUDA implementation not available yet.");
}
void dot_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_WARN("dot_grad_op_cuda: CUDA implementation not available yet.");
}
