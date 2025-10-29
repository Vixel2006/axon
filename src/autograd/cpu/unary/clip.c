#include "autograd/cpu/unary/common.h"

void clip_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("clip_grad_op_cpu: Entering function with n_prev=%d", n_prev);

    if (!out || !out->grad->data->data || !prev)
    {
        LOG_ERROR("clip_grad_op Output tensor, output gradient, or previous "
                  "tensors array is NULL! out=%p, out->grad=%p, prev=%p",
                  (void*) out, (void*) out->grad->data->data, (void*) prev);
        return;
    }

    if (n_prev != 1)
    {
        LOG_ERROR("clip_grad_op Invalid number of previous tensors: %d. "
                  "Expected 1.",
                  n_prev);
        return;
    }

    if (!prev[0])
    {
        LOG_ERROR("clip_grad_op Previous tensor is NULL! prev[0]=%p", (void*) prev[0]);
        return;
    }

    Tensor* a = prev[0];

    if (a->requires_grad && !a->grad->data->data)
    {
        LOG_ERROR("clip_grad_op Tensor 'a' requires grad but its grad is NULL!");
        return;
    }

    ClipExtras* clip_extras = (ClipExtras*) extras;
    float min_val = clip_extras->min_val;
    float max_val = clip_extras->max_val;

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
                float original_val = a->data->data[a_offset];
                LOG_INFO(
                    "clip_grad_op: linear=%d, original_val=%f, min_val=%f, max_val=%f, out_grad=%f",
                    linear, original_val, min_val, max_val, out->grad->data[out_offset]);
                if (original_val >= min_val && original_val <= max_val)
                {
                    a->grad->data->data[a_offset] += out->grad->data->data[out_offset];
                    LOG_INFO("clip_grad_op: a->grad[%d] updated to %f", a_offset,
                             a->grad->data->data[a_offset]);
                }
                else
                {
                    LOG_INFO("clip_grad_op: a->grad[%d] not updated (original_val out of range)",
                             a_offset);
                }
            }
        }
    }
    else
    {
        int i = 0;
        __m256 v_min = _mm256_set1_ps(min_val);
        __m256 v_max = _mm256_set1_ps(max_val);

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
        {
            __m256 original_vals = _mm256_loadu_ps(a->data->data + i);
            __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
            __m256 da = _mm256_loadu_ps(a->grad->data->data + i);

            __m256 lower_bound_mask = _mm256_cmp_ps(original_vals, v_min, _CMP_GE_OQ);
            __m256 upper_bound_mask = _mm256_cmp_ps(original_vals, v_max, _CMP_LE_OQ);
            __m256 within_range_mask = _mm256_and_ps(lower_bound_mask, upper_bound_mask);

            __m256 contrib = _mm256_and_ps(dout, within_range_mask);

            da = _mm256_add_ps(da, contrib);
            _mm256_storeu_ps(a->grad->data->data + i, da);
        }

        for (; i < size; ++i)
        {
            float original_val = a->data->data[i];
            if (original_val >= min_val && original_val <= max_val)
            {
                a->grad->data->data[i] += out->grad->data->data[i];
            }
        }
    }
    LOG_INFO("clip_grad_op_cpu: Exiting function.", n_prev);
}
