#include "autograd/cuda/reduction/common.cuh"
#include "autograd/cuda/reduction/reduction_ops_cuda.h"
#include "utils/indexing.cuh"

__global__ void sum_grad_kernel(const float* out_grad, float* in_grad, const int* shape, int ndim,
                                int axis, int n, const int* in_grad_shape,
                                const int* in_grad_strides, int in_grad_ndim)
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
    int in_grad_idx = get_idx(in_grad_shape, in_grad_strides, in_grad_ndim, idx);
    in_grad[in_grad_idx] += out_grad[out_offset];
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
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (a->requires_grad)
    {
        assert(a->grad && "Input tensor 'a' gradient cannot be NULL if requires_grad");
        assert(a->grad->data && "Input tensor 'a' gradient data cannot be NULL if requires_grad");
        assert(a->grad->data->data &&
               "Input tensor 'a' gradient data pointer cannot be NULL if requires_grad");
        sum_grad_kernel<<<num_blocks, num_threads_per_block>>>(
            out->grad->data->data, a->grad->data->data, a->shape, a->ndim, axis, N, a->shape,
            a->strides, a->ndim);
        CHECK_CUDA();
    }
}
