#include "autograd/cuda/binary/common.cuh"
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
                                          const int* shape, const int* strides, int ndim)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int in_idx = get_idx(shape, strides, ndim, i);
        prev_grad[in_idx] -= out_grad[i];
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
                    prev[0]->strides, prev[0]->ndim);
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
                    prev[0]->strides, prev[0]->ndim);
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
                noncontig_sub_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, prev[1]->grad->data->data, N, prev[1]->shape,
                    prev[1]->strides, prev[1]->ndim);
            }
            CHECK_CUDA();
        }
    }
}
