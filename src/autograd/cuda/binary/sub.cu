#include "autograd/cuda/binary/common.cuh"
#include "autograd/cuda/broadcast_utils.cuh"
#include "utils/indexing.cuh"

__global__ void sub_grad_kernel(const float* out_grad, float* prev_grad, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] -= out_grad[i];
    }
}

__global__ void noncontig_sub_grad_kernel(const float* out_grad, float* prev_grad, int n,
                                          const int* prev_shape, const int* prev_strides,
                                          int prev_ndim, const int* out_shape,
                                          const int* out_strides, int out_ndim)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int in_idx = get_broadcasted_input_idx(i, out_shape, out_ndim, prev_shape, prev_strides, prev_ndim);
        atomicAdd(&prev_grad[in_idx], -out_grad[i]);
    }
}

void sub_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("sub_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    int N = numel(out->shape, out->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (n_prev == 1)
    {
        if (prev[0]->requires_grad)
        {

            if (is_contiguous(prev[0]))
            {
                contig_add_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, prev[0]->grad->data->data, N);
            }
            else
            {
                noncontig_add_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, prev[0]->grad->data->data, N, prev[0]->shape,
                    prev[0]->strides, prev[0]->ndim, out->shape, out->strides, out->ndim);
            }
            CHECK_CUDA();
        }
    }
    else
    {
        if (prev[0]->requires_grad)
        {
            if (is_contiguous(prev[0]))
            {
                contig_add_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, prev[0]->grad->data->data, N);
            }
            else
            {
                noncontig_add_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, prev[0]->grad->data->data, N, prev[0]->shape,
                    prev[0]->strides, prev[0]->ndim, out->shape, out->strides, out->ndim);
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
                sub_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, prev[1]->grad->data->data, N);
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

                noncontig_sub_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, prev[1]->grad->data->data, N, prev[1]->shape,
                    prev[1]->strides, prev[1]->ndim, d_out_shape, d_out_strides, out->ndim);

                cudaFree(d_out_shape);
                cudaFree(d_out_strides);
            }
            CHECK_CUDA();
        }
    }
}
