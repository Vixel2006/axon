#include "logger.h"
#include "optimizers/optimizers.h"

#include "cuda_utils.h"
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

__global__ void sgd_kernel_contig(float* param_data, float* param_grad, float lr, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        param_data[i] -= lr * param_grad[i];
    }
}

__global__ void sgd_kernel_noncontig(float* param_data, float* param_grad, float lr, int n,
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
        param_data[data_idx] -= lr * param_grad[data_idx];
    }
}

void sgd_cuda(Tensor** params, int num_params, float lr)
{
    LOG_INFO("sgd_cuda: Applying SGD optimization (CUDA)");

    for (int i = 0; i < num_params; ++i)
    {
        int N = numel(params[i]->shape, params[i]->ndim);
        int num_threads_per_block = 256;
        int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;
        if (is_contiguous(params[i]))
        {
            sgd_kernel_contig<<<num_blocks, num_threads_per_block>>>(
                params[i]->data->data, params[i]->grad->data->data, lr, N);
            CHECK_CUDA(cudaGetLastError());
        }
        else
        {
            sgd_kernel_noncontig<<<num_blocks, num_threads_per_block>>>(
                params[i]->data->data, params[i]->grad->data->data, lr, N,
                params[i]->shape, params[i]->strides, params[i]->ndim);
            CHECK_CUDA(cudaGetLastError());
        }
    }

    LOG_INFO("sgd_cuda: SGD optimization done successfully.");
}
