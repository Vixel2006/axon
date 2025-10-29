#include "autograd/cpu/binary/common.h"

void rsub_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("rsub_grad_op_cpu: Entering function with n_prev=%d", n_prev);

    // Error checking for null tensors and invalid n_prev
    if (!out || !out->grad->data->data || !prev)
    {
        LOG_ERROR("rsub_grad_op: Output tensor, output gradient, or previous tensors array is "
                  "NULL! out=%p, out->grad=%p, prev=%p",
                  (void*) out, (void*) out->grad->data->data, (void*) prev);
        assert(0 &&
               "rsub_grad_op: Output tensor, output gradient, or previous tensors array is NULL!");
    }

    if (n_prev != 1)
    {
        LOG_ERROR("rsub_grad_op: Invalid number of previous tensors: %d. Expected 1.", n_prev);
        assert(0 && "rsub_grad_op: Invalid number of previous tensors. Expected 1.");
    }

    if (!prev[0])
    {
        LOG_ERROR("rsub_grad_op: Previous tensor is NULL! prev[0]=%p", (void*) prev[0]);
        assert(0 && "rsub_grad_op: Previous tensor is NULL!");
    }

    if (!extras)
    {
        LOG_ERROR("rsub_grad_op: Extras is NULL (scalar value missing)!");
        assert(0 && "rsub_grad_op: Extras is NULL (scalar value missing)!");
    }

    Tensor* a = prev[0];
    float b = *((float*) extras);

    int size = numel(out->shape, out->ndim);

    if (a->requires_grad)
    {
        if (!is_contiguous(a) || !is_contiguous(out) ||
            !shapes_equal(a->shape, a->ndim, out->shape, out->ndim))
        {
            unary_grad_noncontig(out, a, b, unary_rsub_da);
        }
        else
        {
            int i = 0;
            for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
            {
                __m256 a_grad = _mm256_loadu_ps(a->grad->data->data + i);
                __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
                __m256 da = _mm256_sub_ps(a_grad, dout);
                _mm256_storeu_ps(a->grad->data->data + i, da);
            }

            for (; i < size; ++i)
            {
                a->grad->data->data[i] -= out->grad->data->data[i];
            }
        }
    }
    LOG_INFO("rsub_grad_op_cpu: Exiting function.");
}
