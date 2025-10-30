#include "autograd/cuda/reduction/common.cuh"
#include "autograd/cuda/reduction/reduction_ops_cuda.h"
#include "utils/indexing.cuh"

__global__ void sum_full_grad_kernel(float* in_grad_data, float* output_grad, int in_size,
                                     const int* in_grad_shape, const int* in_grad_strides,
                                     int in_grad_ndim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < in_size; i += stride)
    {
        int in_grad_idx = get_idx(in_grad_shape, in_grad_strides, in_grad_ndim, i);
        in_grad_data[in_grad_idx] += output_grad[0];
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
        sum_full_grad_kernel<<<num_blocks, num_threads_per_block>>>(
            a->grad->data->data, out->grad->data->data, N, a->shape, a->strides, a->ndim);

        CHECK_CUDA();
    }
}
