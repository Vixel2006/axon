#include "autograd/cuda/binary/common.cuh"
#include "utils/indexing.cuh"

void rsub_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("rsub_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && "Output tensor cannot be NULL");
    assert(out->grad && "Output tensor gradient cannot be NULL");
    assert(out->grad->data && "Output tensor gradient data cannot be NULL");
    assert(out->grad->data->data && "Output tensor gradient data pointer cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");
    assert(n_prev == 1 && "n_prev must be 1 for rsub_grad_op_cuda");
    assert(prev[0] && "Previous tensor 0 cannot be NULL");
    assert(prev[0]->data && "Previous tensor 0 data cannot be NULL");
    assert(prev[0]->data->data && "Previous tensor 0 data pointer cannot be NULL");
    assert(extras && "Extras (scalar value) cannot be NULL");

    int N = numel(out->shape, out->ndim);
    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (prev[0]->requires_grad)
    {
        assert(prev[0]->grad && "Previous tensor 0 gradient cannot be NULL if requires_grad");
        assert(prev[0]->grad->data &&
               "Previous tensor 0 gradient data cannot be NULL if requires_grad");
        assert(prev[0]->grad->data->data &&
               "Previous tensor 0 gradient data pointer cannot be NULL if requires_grad");
        if (is_contiguous(prev[0]))
        {
            sub_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data->data,
                                                                   prev[0]->grad->data->data, N);
        }
        else
        {
            noncontig_sub_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data->data, prev[0]->grad->data->data, N, prev[0]->shape,
                prev[0]->strides, prev[0]->ndim);
        }
        CHECK_CUDA();
    }
}
