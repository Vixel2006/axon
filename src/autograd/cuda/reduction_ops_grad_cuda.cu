#include "autograd/autograd_reduction.h"
#include "cuda_utils.h"
#include "logger.h"

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

typedef struct
{
    int axis;
} ReductionExtras;

__global__ void sum_full_grad_kernel(float* in_grad_data, float* output_grad, int in_size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < in_size; i += stride)
    {
        in_grad_data[i] += output_grad[0];
    }
}

__global__ void mean_full_grad_kernel(float* in_grad_data, float* output_grad, int in_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < in_size; i += stride)
    {
        in_grad_data[i] += output_grad[0] * (1.0f / in_size);
    }
}

__global__ void max_full_grad_kernel(float* in_grad_data, float* in_data, float* output_grad,
                                     int in_size, float* max)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < in_size; i += stride)
    {
        in_grad_data[i] += (float) (in_data[i] == max[0]) * output_grad[0];
    }
}

__global__ void sum_grad_kernel(const float* out_grad, float* in_grad, const int* shape, int ndim,
                                int axis, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int coords[8];
    int tmp = idx;

    for (int d = ndim - 1; d >= 0; --d)
    {
        coords[d] = tmp % shape[d];
        tmp /= shape[d];
    }

    int out_offset = 0;
    int strides = 1;

    for (int d = ndim - 1; d >= 0; --d)
    {
        if (d == axis) continue;
        out_offset += coords[d] * strides;
        strides *= shape[d];
    }
    in_grad[idx] = out_grad[out_offset];
}

__global__ void mean_grad_kernel(const float* out_grad, float* in_grad, const int* shape, int ndim,
                                 int axis, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int coords[8];
    int tmp = idx;

    for (int d = ndim - 1; d >= 0; --d)
    {
        coords[d] = tmp % shape[d];
        tmp /= shape[d];
    }

    int out_offset = 0;
    int strides = 1;

    for (int d = ndim - 1; d >= 0; --d)
    {
        if (d == axis) continue;
        out_offset += coords[d] * strides;
        strides *= shape[d];
    }
    in_grad[idx] = out_grad[out_offset] / shape[axis];
}

__global__ void max_grad_kernel(const float* out_grad, float* in_grad, const float* in_data,
                                const float* out_data, const int* shape, const int* in_strides,
                                const int* out_strides, int in_ndim, int out_ndim, int reduced_dim,
                                int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int in_coords[8];
        int tmp = i;
        for (int d = in_ndim - 1; d >= 0; --d)
        {
            in_coords[d] = tmp % shape[d];
            tmp /= shape[d];
        }

        int in_offset = 0;
        for (int d = 0; d < in_ndim; ++d)
        {
            in_offset += in_coords[d] * in_strides[d];
        }

        int out_offset = 0;
        for (int d = 0; d < in_ndim; ++d)
        {
            if (d != reduced_dim)
            {
                int out_coord = in_coords[d];
                int out_d = d;
                if (d > reduced_dim && out_ndim < in_ndim)
                {
                    out_d = d - 1;
                }
                if (out_d < out_ndim)
                {
                    out_offset += out_coord * out_strides[out_d];
                }
            }
        }

        if (in_data[in_offset] == out_data[out_offset])
        {
            in_grad[in_offset] += out_grad[out_offset];
        }
    }
}

void sum_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("sum_grad_op_cuda: CUDA implementation called.");
    Tensor* a = prev[0];
    int N = numel(a->shape, a->ndim);

    ReductionExtras* reduction_extras = (ReductionExtras*) extras;
    int axis = reduction_extras->axis;

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / N;

    sum_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data, a->grad->data, a->shape,
                                                           a->ndim, axis, N);

    CHECK_CUDA();

    LOG_INFO("sum_grad_op_cuda: CUDA implementation finished successfully.");
}

void mean_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("mean_grad_op_cuda: CUDA implementation called.");
    Tensor* a = prev[0];
    int N = numel(a->shape, a->ndim);

    ReductionExtras* reduction_extras = (ReductionExtras*) extras;
    int axis = reduction_extras->axis;

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / N;

    mean_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data, a->grad->data,
                                                            a->shape, a->ndim, axis, N);

    CHECK_CUDA();

    LOG_INFO("mean_grad_op_cuda: CUDA implementation finished successfully.");
}

void max_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("max_grad_op_cuda: CUDA implementation called.");

    Tensor* a = prev[0];
    int N = numel(a->shape, a->ndim);

    ReductionExtras* reduction_extras = (ReductionExtras*) extras;
    int axis = reduction_extras->axis;

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    max_grad_kernel<<<num_blocks, num_threads_per_block>>>(
        out->grad->data, a->grad->data, a->data->data, out->data->data, a->shape, a->strides,
        out->strides, a->ndim, out->ndim, axis, N);

    CHECK_CUDA();

    LOG_INFO("max_grad_op_cuda: CUDA implementation finished successfully.");
}

void sum_full_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("full_sum_grad_op_cuda: CUDA implementation called.");
    Tensor* a = prev[0];
    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    sum_full_grad_kernel<<<num_blocks, num_threads_per_block>>>(a->grad->data, out->grad->data, N);

    CHECK_CUDA();

    LOG_INFO("full_sum_grad_op_cuda: CUDA implementation finished successfully.");
}

void mean_full_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("full_mean_grad_op_cuda: CUDA implementation called.");
    Tensor* a = prev[0];
    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    mean_full_grad_kernel<<<num_blocks, num_threads_per_block>>>(a->grad->data, out->grad->data, N);

    CHECK_CUDA();

    LOG_INFO("full_mean_grad_op_cuda: CUDA implementation finished successfully.");
}

void max_full_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("full_max_grad_op_cuda: CUDA implementation called.");
    Tensor* a = prev[0];
    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    max_full_grad_kernel<<<num_blocks, num_threads_per_block>>>(
        a->grad->data, a->data->data, out->grad->data, N, out->data->data);

    CHECK_CUDA();

    LOG_INFO("full_max_grad_op_cuda: CUDA implementation finished successfully.");
}
