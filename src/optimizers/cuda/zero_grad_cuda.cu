#include "logger.h"
#include "optimizers/optimizers.h"
#include <cuda_runtime.h>

#define CHECK_CUDA(err)                                                                            \
    do                                                                                             \
    {                                                                                              \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            LOG_ERROR("cuda-runtime error at %s %d: %s", __FILE__, __LINE__,                       \
                      cudaGetErrorString(err));                                                    \
        }                                                                                          \
    } while (0)

__global__ void zero_grad_kernel_contig(float* param_grad, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        param_grad[i] = 0.0f;
    }
}

__global__ void zero_grad_kernel_noncontig(float* param_grad, int n,
                                       const int* shape, const int* strides, int ndim)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int current_k = i;
        int data_idx = 0;
        for (int d = ndim - 1; d >= 0; --d)
        {
            int dim_idx = current_k % shape[d];
            data_idx += dim_idx * strides[d];
            current_k /= shape[d];
        }
        param_grad[data_idx] = 0.0f;
    }
}

void zero_grad_cuda(Tensor** params, int num_parameters)
{
    LOG_INFO("zero_grad_cuda: Zero gradient running on CUDA.");

    for (int i = 0; i < num_parameters; ++i)
    {
        int N = numel(params[i]->shape, params[i]->ndim);
        int num_threads_per_block = 256;
        int num_blocks = (N + num_threads_per_block + 1) / num_threads_per_block;
        if (is_contiguous(params[i]))
        {
            zero_grad_kernel_contig<<<num_blocks, num_threads_per_block>>>(
                params[i]->grad->data->data, N);
            CHECK_CUDA(cudaGetLastError());
        }
        else
        {
            zero_grad_kernel_noncontig<<<num_blocks, num_threads_per_block>>>(
                params[i]->grad->data->data, N, params[i]->shape, params[i]->strides, params[i]->ndim);
            CHECK_CUDA(cudaGetLastError());
        }
    }

    LOG_INFO("zero_grad_cuda: Zero gradient ran successfully");
}
