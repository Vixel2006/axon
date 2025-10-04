#include "logger.h"
#include "ops/binary_ops.h"
#include <cuda_runtime.h>

__global__ void add_kernel(const float* a, const float* b, float* out, const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        out[i] = a[i] + b[i];
    }
}

__global__ void sub_kernel(const float* a, const float* b, float* out, const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        out[i] = a[i] - b[i];
    }
}

__global__ void mul_kernel(const float* a, const float* b, float* out, const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        out[i] = a[i] * b[i];
    }
}

__global__ void div_kernel(const float* a, const float* b, float* out, const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        out[i] = a[i] / b[i];
    }
}

__global__ void pow_kernel(const float* a, const float* b, float* out, const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        out[i] = powf(a[i], b[i]);
    }
}

void add_op_cuda(Tensor* a, Tensor* b, Tensor* out)
{
    LOG_INFO("Add kernel starts");
    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    add_kernel<<<num_blocks, num_threads_per_block>>>(a->data->data, b->data->data, out->data->data,
                                                      N);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        LOG_ERROR("Addition kernel running error");
        return;
    }
    LOG_INFO("Add kernel done successfully");
}

void sub_op_cuda(Tensor* a, Tensor* b, Tensor* out)
{
    LOG_INFO("Sub kernel starts");
    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    sub_kernel<<<num_blocks, num_threads_per_block>>>(a->data->data, b->data->data, out->data->data,
                                                      N);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        LOG_ERROR("Sub kernel running error.");
        return;
    }
    LOG_INFO("Sub kernel done successfully");
}

void mul_op_cuda(Tensor* a, Tensor* b, Tensor* out)
{
    LOG_INFO("Mul kernel starts");
    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    mul_kernel<<<num_blocks, num_threads_per_block>>>(a->data->data, b->data->data, out->data->data,
                                                      N);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        LOG_ERROR("Mul kernel running error.");
        return;
    }
    LOG_INFO("Mul kernel done successfully");
}

void div_op_cuda(Tensor* a, Tensor* b, Tensor* out)
{
    LOG_INFO("Div kernel starts");
    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    div_kernel<<<num_blocks, num_threads_per_block>>>(a->data->data, b->data->data, out->data->data,
                                                      N);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        LOG_ERROR("Div kernel running error.");
        return;
    }
    LOG_INFO("Div kernel done successfully");
}

void matmul_op_cuda(Tensor* a, Tensor* b, Tensor* out, int N, int K, int P)
{
    (void) a;
    (void) b;
    (void) out;
    (void) N;
    (void) K;
    (void) P;
    LOG_WARN("matmul_op_cuda: CUDA implementation not available yet.");
}

void conv2d_op_cuda(Tensor* in, Tensor* kernel, Tensor* out, const int* kernel_size,
                    const int* stride, const int padding)
{
    (void) in;
    (void) kernel;
    (void) out;
    (void) kernel_size;
    (void) stride;
    (void) padding;
    LOG_WARN("conv2d_op_cuda: CUDA implementation not available yet.");
}

void dot_op_cuda(Tensor* a, Tensor* b, Tensor* out)
{
    (void) a;
    (void) b;
    (void) out;
    LOG_WARN("dot_op_cuda: CUDA implementation not available yet.");
}

void pow_op_cuda(Tensor* a, Tensor* b, Tensor* out)
{
    LOG_INFO("Pow kernel starts");
    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    pow_kernel<<<num_blocks, num_threads_per_block>>>(a->data->data, b->data->data, out->data->data,
                                                      N);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        LOG_ERROR("Pow kernel running error.");
        return;
    }
    LOG_INFO("Pow kernel done successfully");
}