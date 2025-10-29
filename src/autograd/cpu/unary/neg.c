#include "autograd/cpu/unary/common.h"

void neg_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("neg_grad_op_cpu: Entering function with n_prev=%d", n_prev);

    // Error checking for null tensors and invalid n_prev
    if (!out || !out->grad->data->data || !prev)
    {
        LOG_ERROR("neg_grad_op Output tensor, output gradient, or previous "
                  "tensors array is NULL! out=%p, out->grad=%p, prev=%p",
                  (void*) out, (void*) out->grad->data->data, (void*) prev);
        return;
    }

    if (n_prev != 1)
    {
        LOG_ERROR("neg_grad_op Invalid number of previous tensors: %d. "
                  "Expected 1.",
                  n_prev);
        return;
    }

    if (!prev[0])
    {
        LOG_ERROR("neg_grad_op Previous tensor is NULL! prev[0]=%p", (void*) prev[0]);
        return;
    }

    Tensor* a = prev[0];

    if (a->requires_grad && !a->grad->data->data)
    {
        LOG_ERROR("neg_grad_op Tensor 'a' requires grad but its grad is NULL!");
        return;
    }

    int size = numel(a->shape, a->ndim);

    if (!is_contiguous(a) || !is_contiguous(out))
    {
        if (a->requires_grad)
        {
            for (int linear = 0; linear < size; ++linear)
            {
                int idx = linear;
                int a_offset = 0, out_offset = 0;

                for (int d = out->ndim - 1; d >= 0; --d)
                {
                    int coord = idx % out->shape[d];
                    idx /= out->shape[d];

                    a_offset += coord * a->strides[d];
                    out_offset += coord * out->strides[d];
                }
                a->grad->data->data[a_offset] += -out->grad->data->data[out_offset];
            }
        }
    }
    else
    {
        int i = 0;

        __m256 neg_one = _mm256_set1_ps(-1.0f);

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
        {
            __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
            __m256 da = _mm256_loadu_ps(a->grad->data->data + i);

            __m256 contrib = _mm256_mul_ps(dout, neg_one);

            da = _mm256_add_ps(da, contrib);

            _mm256_storeu_ps(a->grad->data->data + i, da);
        }

        for (; i < size; ++i)
        {
            a->grad->data->data[i] += -out->grad->data->data[i];
        }
    }
    LOG_INFO("neg_grad_op_cpu: Exiting function.");
}
