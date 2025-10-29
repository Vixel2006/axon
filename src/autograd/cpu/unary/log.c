#include "autograd/cpu/unary/common.h"

void log_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("log_grad_op_cpu: Entering function with n_prev=%d", n_prev);

    if (!out || !out->grad || !out->grad->data || !out->grad->data->data || !prev)
    {
        LOG_ERROR("log_grad_op: Output tensor, output gradient, or previous "
                  "tensors array is NULL! out=%p, out->grad=%p, prev=%p",
                  (void*) out, (void*) out->grad->data->data, (void*) prev);
        return;
    }

    if (n_prev != 1)
    {
        LOG_ERROR("log_grad_op: Invalid number of previous tensors: %d. "
                  "Expected 1.",
                  n_prev);
        return;
    }

    if (!prev[0])
    {
        LOG_ERROR("log_grad_op: Previous tensor is NULL! prev[0]=%p", (void*) prev[0]);
        return;
    }

    Tensor* a = prev[0];

    if (a->requires_grad && !a->grad->data)
    {
        LOG_ERROR("log_grad_op: Tensor 'a' requires grad but its grad is NULL!");
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
                if (a->data->data[a_offset] == 0.0f)
                {
                    LOG_ERROR("log_grad_op: Division by zero in gradient at "
                              "index %d! Input to log was zero.",
                              linear);
                    // Handle this error, e.g., set gradient to 0 or NaN
                    a->grad->data->data[a_offset] += 0.0f; // Or NAN
                }
                else
                {
                    a->grad->data->data[a_offset] +=
                        out->grad->data->data[out_offset] / a->data->data[a_offset];
                }
            }
        }
    }
    else
    {
        int i = 0;
        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
        {
            __m256 va = _mm256_loadu_ps(a->data->data + i);
            __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
            __m256 da = _mm256_loadu_ps(a->grad->data->data + i);

            // Check for zero values in va to avoid division by zero
            __m256 zero = _mm256_setzero_ps();
            __m256 zero_mask = _mm256_cmp_ps(va, zero, _CMP_EQ_OQ);

            // Replace zero values in va with a small epsilon to avoid Inf/NaN from
            // rcp_ps
            __m256 va_safe = _mm256_blendv_ps(va, _mm256_set1_ps(1e-8f), zero_mask);

            __m256 inv = _mm256_rcp_ps(va_safe);
            __m256 contrib = _mm256_mul_ps(dout, inv);

            // If va was originally zero, set contrib to zero for that lane
            contrib = _mm256_blendv_ps(contrib, zero, zero_mask);

            da = _mm256_add_ps(da, contrib);
            _mm256_storeu_ps(a->grad->data->data + i, da);
        }

        for (; i < size; ++i)
        {
            if (a->data->data[i] == 0.0f)
            {
                LOG_ERROR("log_grad_op: Division by zero in gradient at index "
                          "%d! Input to log was zero.",
                          i);
                a->grad->data->data[i] += 0.0f; // Or NAN
            }
            else
            {
                a->grad->data->data[i] += out->grad->data->data[i] / a->data->data[i];
            }
        }
    }
    LOG_INFO("log_grad_op_cpu: Exiting function.");
}
