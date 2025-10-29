#include "autograd/cuda/unary/common.cuh"
#include "utils/indexing.cuh"

__global__ void relu_grad_kernel(const float* out_grad, const float* prev_data, float* prev_grad,
                                 int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += out_grad[i] * (float) (prev_data[i] > 0.0f);
    }
}

__global__ void noncontig_relu_grad_kernel(const float* out_grad, const float* prev_data,
                                           float* prev_grad, int n, const int* shape,
                                           const int* strides, int ndim)
{
    int idx = blockIdx.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int in_idx = get_idx(shape, strides, ndim, i);
        prev_grad[in_idx] += out_grad[i] * (float) (prev_data[in_idx] > 0.0f);
    }
}

void relu_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("relu_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && "Output tensor cannot be NULL");
    assert(out->grad && "Output tensor gradient cannot be NULL");
    assert(out->grad->data && "Output tensor gradient data cannot be NULL");
    assert(out->grad->data->data && "Output tensor gradient data pointer cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");
    assert(n_prev == 1 && "n_prev must be 1 for relu_grad_op_cuda");

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
            relu_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data->data, prev[0]->data->data, prev[0]->grad->data->data, N);
        }
        else
        {
            noncontig_relu_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data->data, prev[0]->data->data, prev[0]->grad->data->data, N,
                prev[0]->shape, prev[0]->strides, prev[0]->ndim);
        }
        CHECK_CUDA();
    }
}
