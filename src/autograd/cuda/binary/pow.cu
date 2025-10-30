#include "autograd/cuda/binary/common.cuh"
#include "utils/indexing.cuh"

__global__ void scalar_pow_grad_kernel(const float* out_grad, float* prev_data, float* prev_grad,
                                       float power, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += power * powf(prev_data[i] + 1e-7f, power - 1) * out_grad[i];
    }
}

__global__ void noncontig_scalar_pow_grad_kernel(const float* out_grad, float* prev_data,
                                                 float* prev_grad, float power, int n,
                                                 const int* shape, const int* strides, int ndim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int in_idx = get_idx(shape, strides, ndim, i);
        prev_grad[in_idx] += power * powf(prev_data[in_idx] + 1e-7f, power - 1) * out_grad[i];
    }
}

__global__ void base_pow_grad_kernel(const float* out_grad, float* base_data, float* base_grad,
                                     float* power_data, float* power_grad, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        base_grad[i] += power_data[i] * powf(base_data[i] + 1e-7f, power_data[i] - 1) * out_grad[i];
    }
}

__global__ void noncontig_base_pow_grad_kernel(const float* out_grad, float* base_data,
                                               float* base_grad, float* power_data, int n,
                                               const int* shape, const int* strides, int ndim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int in_idx = get_idx(shape, strides, ndim, i);
        base_grad[in_idx] += power_data[in_idx] *
                             powf(base_data[in_idx] + 1e-7f, power_data[in_idx] - 1) * out_grad[i];
    }
}

__global__ void exponent_pow_grad_kernel(const float* out_grad, const float* out_data,
                                         float* base_data, float* power_grad, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        power_grad[i] += out_grad[i] * out_data[i] * logf(base_data[i] + 1e-7f);
    }
}

__global__ void noncontig_exponent_pow_grad_kernel(const float* out_grad, const float* out_data,
                                                   float* base_data, float* power_grad, int n,
                                                   const int* shape, const int* strides, int ndim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int in_idx = get_idx(shape, strides, ndim, i);
        power_grad[in_idx] += out_grad[i] * out_data[i] * logf(base_data[in_idx] + 1e-7f);
    }
}

void pow_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("pow_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && "Output tensor cannot be NULL");
    assert(out->grad && "Output tensor gradient cannot be NULL");
    assert(out->grad->data && "Output tensor gradient data cannot be NULL");
    assert(out->grad->data->data && "Output tensor gradient data pointer cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");
    assert((n_prev == 1 || n_prev == 2) && "n_prev must be 1 or 2 for pow_grad_op_cuda");

    int N = numel(out->shape, out->ndim);
    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (n_prev == 1) // prev[0] ** scalar
    {
        assert(extras && "Extras (scalar value) cannot be NULL for scalar power");
        float scalar_power = *((float*) extras);
        assert(prev[0] && "Previous tensor 0 cannot be NULL");
        assert(prev[0]->data && "Previous tensor 0 data cannot be NULL");
        assert(prev[0]->data->data && "Previous tensor 0 data pointer cannot be NULL");
        if (prev[0]->requires_grad)
        {
            assert(prev[0]->grad && "Previous tensor 0 gradient cannot be NULL if requires_grad");
            assert(prev[0]->grad->data &&
                   "Previous tensor 0 gradient data cannot be NULL if requires_grad");
            assert(prev[0]->grad->data->data &&
                   "Previous tensor 0 gradient data pointer cannot be NULL if requires_grad");
            if (is_contiguous(prev[0]))
            {
                scalar_pow_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, prev[0]->data->data, prev[0]->grad->data->data,
                    scalar_power, N);
            }
            else
            {
                noncontig_scalar_pow_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, prev[0]->data->data, prev[0]->grad->data->data,
                    scalar_power, N, prev[0]->shape, prev[0]->strides, prev[0]->ndim);
            }
            CHECK_CUDA();
        }
    }
    else // base ** power
    {
        assert(prev[0] && "Previous tensor 0 (base) cannot be NULL");
        assert(prev[0]->data && "Previous tensor 0 (base) data cannot be NULL");
        assert(prev[0]->data->data && "Previous tensor 0 (base) data pointer cannot be NULL");
        assert(prev[1] && "Previous tensor 1 (power) cannot be NULL");
        assert(prev[1]->data && "Previous tensor 1 (power) data cannot be NULL");
        assert(prev[1]->data->data && "Previous tensor 1 (power) data pointer cannot be NULL");

        if (prev[0]->requires_grad) // gradient for base
        {
            assert(prev[0]->grad && "Previous tensor 0 gradient cannot be NULL if requires_grad");
            assert(prev[0]->grad->data &&
                   "Previous tensor 0 gradient data cannot be NULL if requires_grad");
            assert(prev[0]->grad->data->data &&
                   "Previous tensor 0 gradient data pointer cannot be NULL if requires_grad");
            if (is_contiguous(prev[0]))
            {
                base_pow_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, prev[0]->data->data, prev[0]->grad->data->data,
                    prev[1]->data->data, prev[1]->data->data, N);
            }
            else
            {
                noncontig_base_pow_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, prev[0]->data->data, prev[0]->grad->data->data,
                    prev[1]->data->data, N, prev[0]->shape, prev[0]->strides, prev[0]->ndim);
            }
            CHECK_CUDA();
        }

        if (prev[1]->requires_grad) // gradient for power
        {
            assert(prev[1]->grad && "Previous tensor 1 gradient cannot be NULL if requires_grad");
            assert(prev[1]->grad->data &&
                   "Previous tensor 1 gradient data cannot be NULL if requires_grad");
            assert(prev[1]->grad->data->data &&
                   "Previous tensor 1 gradient data pointer cannot be NULL if requires_grad");
            if (is_contiguous(prev[1]))
            {
                exponent_pow_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, out->data->data, prev[0]->data->data,
                    prev[1]->grad->data->data, N);
            }
            else
            {
                noncontig_exponent_pow_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, out->data->data, prev[0]->data->data,
                    prev[1]->grad->data->data, N, prev[1]->shape, prev[1]->strides, prev[1]->ndim);
            }
            CHECK_CUDA();
        }
    }
}
