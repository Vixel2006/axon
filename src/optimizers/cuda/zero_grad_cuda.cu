#include "logger.h"
#include "optimizers/optimizers.h"
#include "utils/indexing.cuh"
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

__global__ void zero_grad_kernel_noncontig(float* param_grad, int n, const int* shape,
                                           const int* strides, int ndim)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int data_idx = get_idx(shape, strides, ndim, i);

        param_grad[data_idx] = 0.0f;
    }
}

void zero_grad_cuda(Tensor** params, int num_parameters)
{
    LOG_INFO("zero_grad_cuda: Entering function with num_parameters=%d", num_parameters);

    for (int i = 0; i < num_parameters; ++i)
    {
        if (!params[i] || !params[i]->requires_grad)
        {
            continue;
        }

        if (!params[i]->grad || !params[i]->grad->data || !params[i]->grad->data->data)
        {
            LOG_WARN("zero_grad_cuda: Parameter %d requires grad but its grad tensor or data is "
                     "NULL. Skipping.",
                     i);
            continue;
        }

        int N = numel(params[i]->shape, params[i]->ndim);
        int num_threads_per_block = 256;
        int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;
        if (is_contiguous(params[i]))
        {
            zero_grad_kernel_contig<<<num_blocks, num_threads_per_block>>>(
                params[i]->grad->data->data, N);
            CHECK_CUDA(cudaGetLastError());
        }
        else
        {
            zero_grad_kernel_noncontig<<<num_blocks, num_threads_per_block>>>(
                params[i]->grad->data->data, N, params[i]->shape, params[i]->strides,
                params[i]->ndim);
            CHECK_CUDA(cudaGetLastError());
        }
    }
}
