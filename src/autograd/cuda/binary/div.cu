#include "autograd/cuda/binary/common.cuh"
#include "autograd/cuda/broadcast_utils.cuh"
#include "utils/indexing.cuh"

__global__ void numerator_div_grad_kernel(const float* out_grad, float* prev_grad,
                                          const float* denominator, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += out_grad[i] / (denominator[i] + 1e-7f);
    }
}

__global__ void noncontig_numerator_div_grad_kernel(const float* out_grad, float* prev_grad,
                                                    const float* denominator, int n,
                                                    const int* prev_shape, const int* prev_strides,
                                                    int prev_ndim, const int* out_shape,
                                                    const int* out_strides, int out_ndim)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int in_idx =
            get_broadcasted_input_idx(i, out_shape, out_ndim, prev_shape, prev_strides, prev_ndim);
        atomicAdd(&prev_grad[in_idx], out_grad[i] / (denominator[in_idx] + 1e-7f));
    }
}

__global__ void denominator_div_grad_kernel(const float* out_grad, const float* out_data,
                                            float* prev_grad, float* denominator, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] -= out_data[i] * out_grad[i] / (denominator[i] + 1e-7f);
    }
}

__global__ void noncontig_denominator_div_grad_kernel(const float* out_grad, const float* out_data,
                                                      float* prev_grad, float* denominator, int n,
                                                      const int* prev_shape,
                                                      const int* prev_strides, int prev_ndim,
                                                      const int* out_shape, const int* out_strides,
                                                      int out_ndim)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int in_idx =
            get_broadcasted_input_idx(i, out_shape, out_ndim, prev_shape, prev_strides, prev_ndim);
        atomicAdd(&prev_grad[in_idx], -out_data[i] * out_grad[i] / (denominator[in_idx] + 1e-7f));
    }
}

__global__ void scalar_div_grad_kernel(const float* out_grad, float* prev_grad,
                                       float scalar_denominator, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += out_grad[i] / (scalar_denominator + 1e-7f);
    }
}

void div_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("div_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && "Output tensor cannot be NULL");
    assert(out->grad && "Output tensor gradient cannot be NULL");
    assert(out->grad->data && "Output tensor gradient data cannot be NULL");
    assert(out->grad->data->data && "Output tensor gradient data pointer cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");
    assert((n_prev == 1 || n_prev == 2) && "n_prev must be 1 or 2 for div_grad_op_cuda");

    int N = numel(out->shape, out->ndim);
    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (n_prev == 1) // prev[0] / scalar
    {
        assert(extras && "Extras (scalar value) cannot be NULL for scalar division");
        float scalar_denominator = *((float*) extras);
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
            scalar_div_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data->data, prev[0]->grad->data->data, scalar_denominator, N);
            CHECK_CUDA();
        }
    }
    else
    {
        assert(prev[0] && "Previous tensor 0 (numerator) cannot be NULL");
        assert(prev[0]->data && "Previous tensor 0 (numerator) data cannot be NULL");
        assert(prev[0]->data->data && "Previous tensor 0 (numerator) data pointer cannot be NULL");
        assert(prev[1] && "Previous tensor 1 (denominator) cannot be NULL");
        assert(prev[1]->data && "Previous tensor 1 (denominator) data cannot be NULL");
        assert(prev[1]->data->data &&
               "Previous tensor 1 (denominator) data pointer cannot be NULL");

        if (prev[0]->requires_grad)
        {
            assert(prev[0]->grad && "Previous tensor 0 gradient cannot be NULL if requires_grad");
            assert(prev[0]->grad->data &&
                   "Previous tensor 0 gradient data cannot be NULL if requires_grad");
            assert(prev[0]->grad->data->data &&
                   "Previous tensor 0 gradient data pointer cannot be NULL if requires_grad");
            if (is_contiguous(prev[0]))
            {
                numerator_div_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, prev[0]->grad->data->data, prev[1]->data->data, N);
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

                noncontig_numerator_div_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, prev[0]->grad->data->data, prev[1]->data->data, N,
                    prev[0]->shape, prev[0]->strides, prev[0]->ndim, d_out_shape, d_out_strides,
                    out->ndim);

                cudaFree(d_out_shape);
                cudaFree(d_out_strides);
            }
            CHECK_CUDA();
        }

        if (prev[1]->requires_grad)
        {
            assert(prev[1]->grad && "Previous tensor 1 gradient cannot be NULL if requires_grad");
            assert(prev[1]->grad->data &&
                   "Previous tensor 1 gradient data cannot be NULL if requires_grad");
            assert(prev[1]->grad->data->data &&
                   "Previous tensor 1 gradient data pointer cannot be NULL if requires_grad");
            if (is_contiguous(prev[1]))
            {
                denominator_div_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, out->data->data, prev[1]->grad->data->data,
                    prev[1]->data->data, N);
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

                noncontig_denominator_div_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, out->data->data, prev[1]->grad->data->data,
                    prev[1]->data->data, N, prev[1]->shape, prev[1]->strides, prev[1]->ndim,
                    d_out_shape, d_out_strides, out->ndim);

                cudaFree(d_out_shape);
                cudaFree(d_out_strides);
            }
            CHECK_CUDA();
        }
    }
}
