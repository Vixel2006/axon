#include "autograd/cpu/reduction/common.h"

void max_full_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("max_full_grad_op_cpu: Entering function with n_prev=%d", n_prev);

    if (!out || !out->grad || !out->grad->data->data || !prev)
    {
        LOG_ERROR("max_full_grad_op ERROR: Output tensor, output gradient, or previous "
                  "tensors array is NULL! out=%p, out->grad=%p, prev=%p",
                  (void*) out, out ? (void*) out->grad : NULL, (void*) prev);
        return;
    }

    if (n_prev != 1)
    {
        LOG_ERROR("max_full_grad_op ERROR: Invalid number of previous tensors: %d. "
                  "Expected 1.",
                  n_prev);
        return;
    }

    if (!prev[0])
    {
        LOG_ERROR("max_full_grad_op ERROR: Previous tensor is NULL! prev[0]=%p", (void*) prev[0]);
        return;
    }

    Tensor* in = prev[0];

    if (!in->requires_grad)
    {
        return;
    }

    if (!in->grad || !in->grad->data->data)
    {
        LOG_ERROR("max_full_grad_op ERROR: Input tensor requires grad but its grad data is NULL!");
        return;
    }

    float output_grad = out->grad->data->data[0];
    float max_val = out->data->data[0];
    int in_size = numel(in->shape, in->ndim);

    if (is_contiguous(in))
    {
        __m256 grad_vec = _mm256_set1_ps(output_grad);
        __m256 max_vec = _mm256_set1_ps(max_val);

        int i = 0;
        for (; i + SIMD_WIDTH - 1 < in_size; i += SIMD_WIDTH)
        {
            __m256 data_vec = _mm256_loadu_ps(in->data->data + i);
            __m256 mask = _mm256_cmp_ps(data_vec, max_vec, _CMP_EQ_OQ);
            __m256 grad_contrib = _mm256_and_ps(grad_vec, mask);
            __m256 in_grad = _mm256_loadu_ps(in->grad->data->data + i);
            __m256 new_grad = _mm256_add_ps(in_grad, grad_contrib);
            _mm256_storeu_ps(in->grad->data->data + i, new_grad);
        }

        for (; i < in_size; ++i)
        {
            if (in->data->data[i] == max_val)
            {
                in->grad->data->data[i] += output_grad;
            }
        }
    }
    else
    {
        int* in_strides = in->strides;
        int* in_shape = in->shape;
        int in_ndim = in->ndim;

        for (int linear_idx = 0; linear_idx < in_size; ++linear_idx)
        {
            int temp_idx = linear_idx;
            int in_offset = 0;

            for (int d = in_ndim - 1; d >= 0; --d)
            {
                int coord = temp_idx % in_shape[d];
                temp_idx /= in_shape[d];
                in_offset += coord * in_strides[d];
            }

            if (in->data->data[in_offset] == max_val)
            {
                in->grad->data->data[in_offset] += output_grad;
            }
        }
    }
    LOG_INFO("max_full_grad_op_cpu: Exiting function.");
}
