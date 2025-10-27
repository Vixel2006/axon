#include "autograd/autograd_reduction.h"
#include "cuda_utils.h"
#include "logger.h"
#include <assert.h>

#define CHECK_CUDA()                                                                               \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = cudaGetLastError();                                                      \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            LOG_ERROR("CUDA runtime error at %s:%d: %s", __FILE__, __LINE__,                       \
                      cudaGetErrorString(err));                                                    \
            assert(0 && "CUDA runtime error");                                                     \
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

// WARNING: This kernel is very stupid, there is million places it can do warp divergence in.
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
    LOG_INFO("sum_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && "Output tensor cannot be NULL");
    assert(out->grad && "Output tensor gradient cannot be NULL");
    assert(out->grad->data && "Output tensor gradient data cannot be NULL");
    assert(out->grad->data->data && "Output tensor gradient data pointer cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");
    assert(n_prev == 1 && "n_prev must be 1 for sum_grad_op_cuda");
    assert(extras && "Extras (ReductionExtras) cannot be NULL");

    Tensor* a = prev[0];
    assert(a && "Input tensor 'a' cannot be NULL");
    assert(a->data && "Input tensor 'a' data cannot be NULL");
    assert(a->data->data && "Input tensor 'a' data pointer cannot be NULL");
    assert(a->shape && "Input tensor 'a' shape cannot be NULL");

    ReductionExtras* reduction_extras = (ReductionExtras*) extras;
    int axis = reduction_extras->axis;
    assert(axis >= 0 && axis < a->ndim && "Axis is out of bounds for input tensor 'a'");

    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / N;

    if (a->requires_grad)
    {
        assert(a->grad && "Input tensor 'a' gradient cannot be NULL if requires_grad");
        assert(a->grad->data && "Input tensor 'a' gradient data cannot be NULL if requires_grad");
        assert(a->grad->data->data &&
               "Input tensor 'a' gradient data pointer cannot be NULL if requires_grad");
        sum_grad_kernel<<<num_blocks, num_threads_per_block>>>(
            out->grad->data->data, a->grad->data->data, a->shape, a->ndim, axis, N);
        CHECK_CUDA();
    }
}

void mean_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("mean_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && "Output tensor cannot be NULL");
    assert(out->grad && "Output tensor gradient cannot be NULL");
    assert(out->grad->data && "Output tensor gradient data cannot be NULL");
    assert(out->grad->data->data && "Output tensor gradient data pointer cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");
    assert(n_prev == 1 && "n_prev must be 1 for mean_grad_op_cuda");
    assert(extras && "Extras (ReductionExtras) cannot be NULL");

    Tensor* a = prev[0];
    assert(a && "Input tensor 'a' cannot be NULL");
    assert(a->data && "Input tensor 'a' data cannot be NULL");
    assert(a->data->data && "Input tensor 'a' data pointer cannot be NULL");
    assert(a->shape && "Input tensor 'a' shape cannot be NULL");

    ReductionExtras* reduction_extras = (ReductionExtras*) extras;
    int axis = reduction_extras->axis;
    assert(axis >= 0 && axis < a->ndim && "Axis is out of bounds for input tensor 'a'");

    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / N;

    if (a->requires_grad)
    {
        assert(a->grad && "Input tensor 'a' gradient cannot be NULL if requires_grad");
        assert(a->grad->data && "Input tensor 'a' gradient data cannot be NULL if requires_grad");
        assert(a->grad->data->data &&
               "Input tensor 'a' gradient data pointer cannot be NULL if requires_grad");
        mean_grad_kernel<<<num_blocks, num_threads_per_block>>>(
            out->grad->data->data, a->grad->data->data, a->shape, a->ndim, axis, N);
        CHECK_CUDA();
    }
}

void max_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("max_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && "Output tensor cannot be NULL");
    assert(out->grad && "Output tensor gradient cannot be NULL");
    assert(out->grad->data && "Output tensor gradient data cannot be NULL");
    assert(out->grad->data->data && "Output tensor gradient data pointer cannot be NULL");
    assert(out->data && "Output tensor data cannot be NULL");
    assert(out->data->data && "Output tensor data pointer cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");
    assert(n_prev == 1 && "n_prev must be 1 for max_grad_op_cuda");
    assert(extras && "Extras (ReductionExtras) cannot be NULL");

    Tensor* a = prev[0];
    assert(a && "Input tensor 'a' cannot be NULL");
    assert(a->data && "Input tensor 'a' data cannot be NULL");
    assert(a->data->data && "Input tensor 'a' data pointer cannot be NULL");
    assert(a->shape && "Input tensor 'a' shape cannot be NULL");
    assert(a->strides && "Input tensor 'a' strides cannot be NULL");

    ReductionExtras* reduction_extras = (ReductionExtras*) extras;
    int axis = reduction_extras->axis;
    assert(axis >= 0 && axis < a->ndim && "Axis is out of bounds for input tensor 'a'");

    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (a->requires_grad)
    {
        assert(a->grad && "Input tensor 'a' gradient cannot be NULL if requires_grad");
        assert(a->grad->data && "Input tensor 'a' gradient data cannot be NULL if requires_grad");
        assert(a->grad->data->data &&
               "Input tensor 'a' gradient data pointer cannot be NULL if requires_grad");
        max_grad_kernel<<<num_blocks, num_threads_per_block>>>(
            out->grad->data->data, a->grad->data->data, a->data->data, out->data->data, a->shape,
            a->strides, out->strides, a->ndim, out->ndim, axis, N);

        CHECK_CUDA();
    }
}

void sum_full_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("sum_full_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && "Output tensor cannot be NULL");
    assert(out->grad && "Output tensor gradient cannot be NULL");
    assert(out->grad->data && "Output tensor gradient data cannot be NULL");
    assert(out->grad->data->data && "Output tensor gradient data pointer cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");
    assert(n_prev == 1 && "n_prev must be 1 for sum_full_grad_op_cuda");

    Tensor* a = prev[0];
    assert(a && "Input tensor 'a' cannot be NULL");
    assert(a->data && "Input tensor 'a' data cannot be NULL");
    assert(a->data->data && "Input tensor 'a' data pointer cannot be NULL");

    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (a->requires_grad)
    {
        assert(a->grad && "Input tensor 'a' gradient cannot be NULL if requires_grad");
        assert(a->grad->data && "Input tensor 'a' gradient data cannot be NULL if requires_grad");
        assert(a->grad->data->data &&
               "Input tensor 'a' gradient data pointer cannot be NULL if requires_grad");
        sum_full_grad_kernel<<<num_blocks, num_threads_per_block>>>(a->grad->data->data,
                                                                    out->grad->data->data, N);

        CHECK_CUDA();
    }
}

void mean_full_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("mean_full_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && "Output tensor cannot be NULL");
    assert(out->grad && "Output tensor gradient cannot be NULL");
    assert(out->grad->data && "Output tensor gradient data cannot be NULL");
    assert(out->grad->data->data && "Output tensor gradient data pointer cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");
    assert(n_prev == 1 && "n_prev must be 1 for mean_full_grad_op_cuda");

    Tensor* a = prev[0];
    assert(a && "Input tensor 'a' cannot be NULL");
    assert(a->data && "Input tensor 'a' data cannot be NULL");
    assert(a->data->data && "Input tensor 'a' data pointer cannot be NULL");

    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (a->requires_grad)
    {
        assert(a->grad && "Input tensor 'a' gradient cannot be NULL if requires_grad");
        assert(a->grad->data && "Input tensor 'a' gradient data cannot be NULL if requires_grad");
        assert(a->grad->data->data &&
               "Input tensor 'a' gradient data pointer cannot be NULL if requires_grad");
        mean_full_grad_kernel<<<num_blocks, num_threads_per_block>>>(a->grad->data->data,
                                                                     out->grad->data->data, N);

        CHECK_CUDA();
    }
}

void max_full_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("max_full_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && "Output tensor cannot be NULL");
    assert(out->grad && "Output tensor gradient cannot be NULL");
    assert(out->grad->data && "Output tensor gradient data cannot be NULL");
    assert(out->grad->data->data && "Output tensor gradient data pointer cannot be NULL");
    assert(out->data && "Output tensor data cannot be NULL");
    assert(out->data->data && "Output tensor data pointer cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");
    assert(n_prev == 1 && "n_prev must be 1 for max_full_grad_op_cuda");

    Tensor* a = prev[0];
    assert(a && "Input tensor 'a' cannot be NULL");
    assert(a->data && "Input tensor 'a' data cannot be NULL");
    assert(a->data->data && "Input tensor 'a' data pointer cannot be NULL");

    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (a->requires_grad)
    {
        assert(a->grad && "Input tensor 'a' gradient cannot be NULL if requires_grad");
        assert(a->grad->data && "Input tensor 'a' gradient data cannot be NULL if requires_grad");
        assert(a->grad->data->data &&
               "Input tensor 'a' gradient data pointer cannot be NULL if requires_grad");
        max_full_grad_kernel<<<num_blocks, num_threads_per_block>>>(
            a->grad->data->data, a->data->data, out->grad->data->data, N, out->data->data);

        CHECK_CUDA();
    }
}
