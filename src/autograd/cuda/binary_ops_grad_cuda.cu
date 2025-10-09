#include "autograd/autograd_binary.h"
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

__global__ void add_grad_kernel(const float* out_grad, float* prev_grad, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += out_grad[i];
    }
}

__global__ void sub_grad_kernel(const float* out_grad, float* prev_grad, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] -= out_grad[i];
    }
}

__global__ void mul_grad_kernel(const float* out_grad, float* prev_grad, float* other_data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += out_grad[i] * other_data[i];
    }
}

__global__ void scalar_mul_grad_kernel(const float* out_grad, float* prev_grad, float scalar, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += out_grad[i] * scalar;
    }
}

__global__ void scalar_pow_grad_kernel(const float* out_grad, float* prev_data, float* prev_grad,
                                       float power, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += power * powf(prev_data[i], power - 1) * out_grad[i];
    }
}

__global__ void base_pow_grad_kernel(const float* out_grad, float* base_data, float* base_grad,
                                     float* power_data, float* power_grad, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        base_grad[i] += power_data[i] * powf(base_data[i], power_data[i] - 1) * out_grad[i];
    }
}

__global__ void exponent_pow_grad_kernel(const float* out_grad, const float* out_data,
                                         float* base_data, float* power_grad, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        power_grad[i] += out_data[i] * logf(base_data[i]);
    }
}

__global__ void numerator_div_grad_kernel(const float* out_grad, float* prev_grad,
                                          const float* denominator, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += out_grad[i] / denominator[i];
    }
}

__global__ void denominator_div_grad_kernel(const float* out_grad, const float* out_data,
                                            float* prev_grad, float* denominator, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] -= out_data[i] * out_grad[i] / denominator[i];
    }
}

__global__ void scalar_div_grad_kernel(const float* out_grad, float* prev_grad,
                                       float scalar_denominator, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += out_grad[i] / scalar_denominator;
    }
}

__global__ void scalar_rdiv_grad_kernel(const float* out_grad, const float* out_data,
                                        float* prev_grad, float scalar_numerator,
                                        const float* prev_data, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] -= out_grad[i] * out_data[i] / prev_data[i];
    }
}

void add_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("add_grad_op_cuda: CUDA implementation called.");

    int N = numel(out->shape, out->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (n_prev == 1)
    {
        if (prev[0]->requires_grad)
        {
            add_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data,
                                                                   prev[0]->grad->data, N);
            CHECK_CUDA();
        }
    }
    else
    {
        if (prev[0]->requires_grad)
        {
            add_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data,
                                                                   prev[0]->grad->data, N);
            CHECK_CUDA();
        }

        if (prev[1]->requires_grad)
        {
            add_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data,
                                                                   prev[1]->grad->data, N);
            CHECK_CUDA();
        }
    }
    LOG_INFO("add_grad_op_cuda: CUDA implementation finished successfully.");
}

void sub_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("sub_grad_op_cuda: CUDA implementation called.");
    int N = numel(out->shape, out->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (n_prev == 1)
    {
        if (prev[0]->requires_grad)
        {
            add_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data,
                                                                   prev[0]->grad->data, N);
            CHECK_CUDA();
        }
    }
    else
    {
        if (prev[0]->requires_grad)
        {
            add_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data,
                                                                   prev[0]->grad->data, N);
            CHECK_CUDA();
        }

        if (prev[1]->requires_grad)
        {
            sub_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data,
                                                                   prev[1]->grad->data, N);
            CHECK_CUDA();
        }
    }

    LOG_INFO("sub_grad_op_cuda: CUDA implementation finished successfully.");
}

void rsub_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{

    LOG_INFO("rsub_grad_op_cuda: CUDA implementation called.");
    int N = numel(out->shape, out->ndim);
    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (prev[0]->requires_grad)
    {
        sub_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data, prev[0]->grad->data,
                                                               N);
        CHECK_CUDA();
    }

    LOG_INFO("rsub_grad_op_cuda: CUDA implementation finished successfully.");
}

void mul_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("mul_grad_op_cuda: CUDA implementation called.");
    int N = numel(out->shape, out->ndim);
    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (n_prev == 1)
    {
        float* scalar = (float*) extras;
        if (prev[0]->requires_grad)
        {
            scalar_mul_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data, prev[0]->grad->data, *scalar, N);
            CHECK_CUDA();
        }
    }
    else
    {
        if (prev[0]->requires_grad)
        {
            mul_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data, prev[0]->grad->data, prev[1]->data->data, N);
            CHECK_CUDA();
        }

        if (prev[1]->requires_grad)
        {
            mul_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data, prev[1]->grad->data, prev[0]->data->data, N);
            CHECK_CUDA();
        }
    }

    LOG_INFO("mul_grad_op_cuda: CUDA implementation finished successfully.");
}
void pow_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("pow_grad_op_cuda: CUDA implementation called.");
    int N = numel(out->shape, out->ndim);
    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (n_prev == 1) // prev[0] ** scalar
    {
        float scalar_power = *((float*) extras);
        if (prev[0]->requires_grad)
        {
            scalar_pow_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data, prev[0]->data->data, prev[0]->grad->data, scalar_power, N);
            CHECK_CUDA();
        }
    }
    else // base ** power
    {
        if (prev[0]->requires_grad) // gradient for base
        {
            // NOTE: The 'power_grad' parameter in base_pow_grad_kernel is used as the exponent.
            // This might be a typo in the kernel definition, as it should ideally be 'power_data -
            // 1'. Using prev[1]->data->data for both power_data and power_grad to match the
            // signature.
            base_pow_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data, prev[0]->data->data, prev[0]->grad->data, prev[1]->data->data,
                prev[1]->data->data, N);
            CHECK_CUDA();
        }

        if (prev[1]->requires_grad) // gradient for power
        {
            exponent_pow_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data, out->data->data, prev[0]->data->data, prev[1]->grad->data, N);
            CHECK_CUDA();
        }
    }
    LOG_INFO("pow_grad_op_cuda: CUDA implementation finished successfully.");
}

void div_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("div_grad_op_cuda: CUDA implementation called.");
    int N = numel(out->shape, out->ndim);
    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (n_prev == 1) // prev[0] / scalar
    {
        float scalar_denominator = *((float*) extras);
        if (prev[0]->requires_grad)
        {
            scalar_div_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data, prev[0]->grad->data, scalar_denominator, N);
            CHECK_CUDA();
        }
    }
    else
    {
        if (prev[0]->requires_grad)
        {
            numerator_div_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data, prev[0]->grad->data, prev[1]->data->data, N);
            CHECK_CUDA();
        }

        if (prev[1]->requires_grad)
        {
            denominator_div_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data, out->data->data, prev[1]->grad->data, prev[1]->data->data, N);
            CHECK_CUDA();
        }
    }
    LOG_INFO("div_grad_op_cuda: CUDA implementation finished successfully.");
}
void rdiv_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("rdiv_grad_op_cuda: CUDA implementation called.");
    int N = numel(out->shape, out->ndim);
    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (n_prev == 1) // scalar / prev[0]
    {
        float scalar_numerator = *((float*) extras);
        if (prev[0]->requires_grad)
        {
            scalar_rdiv_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data, out->data->data, prev[0]->grad->data, scalar_numerator,
                prev[0]->data->data, N);
            CHECK_CUDA();
        }
    }
    else
    {
        if (prev[0]->requires_grad)
        {
            denominator_div_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data, out->data->data, prev[0]->grad->data, prev[0]->data->data, N);
            CHECK_CUDA();
        }

        if (prev[1]->requires_grad)
        {
            numerator_div_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data, prev[1]->grad->data, prev[0]->data->data, N);
            CHECK_CUDA();
        }
    }
    LOG_INFO("rdiv_grad_op_cuda: CUDA implementation finished successfully.");
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
