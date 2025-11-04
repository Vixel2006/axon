#include "autograd/cuda/unary/common.cuh"
#include "autograd/cuda/unary/unary_ops_cuda.h"
#include "utils/indexing.cuh"
#include "autograd/cuda/broadcast_utils.cuh"

__global__ void neg_grad_kernel(const float* out_grad, float* prev_grad, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] -= out_grad[i];
    }
}

__global__ void noncontig_neg_grad_kernel(const float* out_grad, float* prev_grad, int n,
                                          const int* prev_shape, const int* prev_strides,
                                          int prev_ndim, const int* out_shape,
                                          const int* out_strides, int out_ndim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int in_idx = get_broadcasted_input_idx(i, out_shape, out_ndim, prev_shape, prev_strides, prev_ndim);
        atomicAdd(&prev_grad[in_idx], -out_grad[i]);
    }
}

void neg_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("neg_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && "Output tensor cannot be NULL");
    assert(out->grad && "Output tensor gradient cannot be NULL");
    assert(out->grad->data && "Output tensor gradient data cannot be NULL");
    assert(out->grad->data->data && "Output tensor gradient data pointer cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");
    assert(n_prev == 1 && "n_prev must be 1 for neg_grad_op_cuda");

    Tensor* a = prev[0];
    assert(a && "Input tensor 'a' cannot be NULL");
    assert(a->data && "Input tensor 'a' data cannot be NULL");
    assert(a->data->data && "Input tensor 'a' data pointer cannot be NULL");

    int N = numel(out->shape, out->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (prev[0]->requires_grad)
    {
        assert(a->grad && "Input tensor 'a' gradient cannot be NULL if requires_grad");
        assert(a->grad->data && "Input tensor 'a' gradient data cannot be NULL if requires_grad");
        assert(a->grad->data->data &&
               "Input tensor 'a' gradient data pointer cannot be NULL if requires_grad");
        if (is_contiguous(prev[0]))
        {
            neg_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data->data,
                                                                   prev[0]->grad->data->data, N);
        }
        else
        {
            int* d_out_shape;
            int* d_out_strides;

            cudaMalloc(&d_out_shape, out->ndim * sizeof(int));
            cudaMalloc(&d_out_strides, out->ndim * sizeof(int));

            cudaMemcpy(d_out_shape, out->shape, out->ndim * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_out_strides, out->strides, out->ndim * sizeof(int), cudaMemcpyHostToDevice);

            noncontig_neg_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data->data, prev[0]->grad->data->data, N, prev[0]->shape,
                prev[0]->strides, prev[0]->ndim, d_out_shape, d_out_strides, out->ndim);

            cudaFree(d_out_shape);
            cudaFree(d_out_strides);
        }
        CHECK_CUDA();
    }
}
