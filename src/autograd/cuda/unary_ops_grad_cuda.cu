#include "autograd/autograd_unary.h"
#include "logger.h"
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA()                                                                               \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = cudaGetLastError();                                                      \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            LOG_ERROR("CUDA runtime error at %s:%d: %s", __FILE__, __LINE__,                       \
                      cudaGetErrorString(err));                                                    \
            return;                                                                                \
        }                                                                                          \
    } while (0)

__global__ void relu_grad_kernel(const float* out_grad, const float* prev_data, float* prev_grad,
                                 int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += out_grad[i] * (float) (prev_data[i] > 0.0f);
    }
}

void relu_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("relu_grad_op_cuda: CUDA implementation called.");
    int N = numel(out->shape, out->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (prev[0]->requires_grad)
    {
        relu_grad_kernel<<<num_blocks, num_threads_per_block>>>(
            out->grad->data, prev[0]->data->data, prev[0]->grad->data, N);
        CHECK_CUDA();
    }
    LOG_INFO("relu_grad_op_cuda: CUDA implementation finished successfully.");
}

__global__ void log_grad_kernel(const float* out_grad, const float* prev_data, float* prev_grad,
                                int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += out_grad[i] / prev_data[i];
    }
}

void log_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("log_grad_op_cuda: CUDA implementation called.");
    int N = numel(out->shape, out->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (prev[0]->requires_grad)
    {
        log_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data, prev[0]->data->data,
                                                               prev[0]->grad->data, N);
        CHECK_CUDA();
    }
    LOG_INFO("log_grad_op_cuda: CUDA implementation finished successfully.");
}
__global__ void exp_grad_kernel(const float* out_grad, const float* out_data, float* prev_grad,
                                int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += out_grad[i] * out_data[i];
    }
}

void exp_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("exp_grad_op_cuda: CUDA implementation called.");
    int N = numel(out->shape, out->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (prev[0]->requires_grad)
    {
        exp_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data, out->data->data,
                                                               prev[0]->grad->data, N);
        CHECK_CUDA();
    }
    LOG_INFO("exp_grad_op_cuda: CUDA implementation finished successfully.");
}
__global__ void abs_grad_kernel(const float* out_grad, const float* prev_data, float* prev_grad,
                                int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        float x = prev_data[i];
        float sign = (x > 0.0f) - (x < 0.0f);
        prev_grad[i] += out_grad[i] * sign;
    }
}

void abs_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("abs_grad_op_cuda: CUDA implementation called.");
    int N = numel(out->shape, out->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (prev[0]->requires_grad)
    {
        abs_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data, prev[0]->data->data,
                                                               prev[0]->grad->data, N);
        CHECK_CUDA();
    }
    LOG_INFO("abs_grad_op_cuda: CUDA implementation finished successfully.");
}
__global__ void neg_grad_kernel(const float* out_grad, float* prev_grad, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] -= out_grad[i];
    }
}

void neg_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("neg_grad_op_cuda: CUDA implementation called.");
    int N = numel(out->shape, out->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (prev[0]->requires_grad)
    {
        neg_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data, prev[0]->grad->data,
                                                               N);
        CHECK_CUDA();
    }
    LOG_INFO("neg_grad_op_cuda: CUDA implementation finished successfully.");
}

__global__ void clip_grad_kernel(const float* out_grad, const float* prev_data, float* prev_grad,
                                 float min_val, float max_val, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        float x = prev_data[i];
        float mask = (prev_data[i] >= min_val) & (x <= max_val);
        prev_grad[i] += out_grad[i] * mask;
    }
}

void clip_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("clip_grad_op_cuda: CUDA implementation called.");
    int N = numel(out->shape, out->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    ClipExtras* clip_extras = (ClipExtras*) extras;
    float min_val = clip_extras->min_val;
    float max_val = clip_extras->max_val;

    if (prev[0]->requires_grad)
    {
        clip_grad_kernel<<<num_blocks, num_threads_per_block>>>(
            out->grad->data, prev[0]->data->data, prev[0]->grad->data, min_val, max_val, N);
        CHECK_CUDA();
    }
    LOG_INFO("clip_grad_op_cuda: CUDA implementation finished successfully.");
}
