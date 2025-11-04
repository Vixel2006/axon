#include "autograd/cuda/binary/common.cuh"
#include "autograd/cuda/broadcast_utils.cuh"
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
                                                 const int* prev_shape, const int* prev_strides,
                                                 int prev_ndim, const int* out_shape,
                                                 const int* out_strides, int out_ndim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int in_idx =
            get_broadcasted_input_idx(i, out_shape, out_ndim, prev_shape, prev_strides, prev_ndim);
        atomicAdd(&prev_grad[in_idx],
                  power * powf(prev_data[in_idx] + 1e-7f, power - 1) * out_grad[i]);
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
                                               const int* base_shape, const int* base_strides,
                                               int base_ndim, const int* out_shape,
                                               const int* out_strides, int out_ndim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int in_idx =
            get_broadcasted_input_idx(i, out_shape, out_ndim, base_shape, base_strides, base_ndim);
        int power_data_idx =
            get_broadcasted_input_idx(i, out_shape, out_ndim, base_shape, base_strides,
                                      base_ndim); // Assuming power_data has same broadcasted shape
                                                  // as base_data for this kernel
        atomicAdd(&base_grad[in_idx],
                  power_data[power_data_idx] *
                      powf(base_data[in_idx] + 1e-7f, power_data[power_data_idx] - 1) *
                      out_grad[i]);
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
                                                   const int* power_shape, const int* power_strides,
                                                   int power_ndim, const int* out_shape,
                                                   const int* out_strides, int out_ndim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int in_idx = get_broadcasted_input_idx(i, out_shape, out_ndim, power_shape, power_strides,
                                               power_ndim);
        int base_data_idx =
            get_broadcasted_input_idx(i, out_shape, out_ndim, power_shape, power_strides,
                                      power_ndim); // Assuming base_data has same broadcasted shape
                                                   // as power_data for this kernel
        atomicAdd(&power_grad[in_idx],
                  out_grad[i] * out_data[i] * logf(base_data[base_data_idx] + 1e-7f));
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
                int* d_out_shape;
                int* d_out_strides;

                cudaMalloc(&d_out_shape, out->ndim * sizeof(int));
                cudaMalloc(&d_out_strides, out->ndim * sizeof(int));

                cudaMemcpy(d_out_shape, out->shape, out->ndim * sizeof(int),
                           cudaMemcpyHostToDevice);
                cudaMemcpy(d_out_strides, out->strides, out->ndim * sizeof(int),
                           cudaMemcpyHostToDevice);

                noncontig_scalar_pow_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, prev[0]->data->data, prev[0]->grad->data->data,
                    scalar_power, N, prev[0]->shape, prev[0]->strides, prev[0]->ndim, d_out_shape,
                    d_out_strides, out->ndim);

                cudaFree(d_out_shape);
                cudaFree(d_out_strides);
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
                int* d_out_shape;
                int* d_out_strides;

                cudaMalloc(&d_out_shape, out->ndim * sizeof(int));
                cudaMalloc(&d_out_strides, out->ndim * sizeof(int));

                cudaMemcpy(d_out_shape, out->shape, out->ndim * sizeof(int),
                           cudaMemcpyHostToDevice);
                cudaMemcpy(d_out_strides, out->strides, out->ndim * sizeof(int),
                           cudaMemcpyHostToDevice);

                noncontig_base_pow_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, prev[0]->data->data, prev[0]->grad->data->data,
                    prev[1]->data->data, N, prev[0]->shape, prev[0]->strides, prev[0]->ndim,
                    d_out_shape, d_out_strides, out->ndim);

                cudaFree(d_out_shape);
                cudaFree(d_out_strides);
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
                int* d_out_shape;
                int* d_out_strides;

                cudaMalloc(&d_out_shape, out->ndim * sizeof(int));
                cudaMalloc(&d_out_strides, out->ndim * sizeof(int));

                cudaMemcpy(d_out_shape, out->shape, out->ndim * sizeof(int),
                           cudaMemcpyHostToDevice);
                cudaMemcpy(d_out_strides, out->strides, out->ndim * sizeof(int),
                           cudaMemcpyHostToDevice);

                noncontig_exponent_pow_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, out->data->data, prev[0]->data->data,
                    prev[1]->grad->data->data, N, prev[1]->shape, prev[1]->strides, prev[1]->ndim,
                    d_out_shape, d_out_strides, out->ndim);

                cudaFree(d_out_shape);
                cudaFree(d_out_strides);
            }
            CHECK_CUDA();
        }
    }
}
