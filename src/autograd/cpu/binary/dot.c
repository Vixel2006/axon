#include "autograd/cpu/binary/common.h"

void dot_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("dot_grad_op_cpu: Entering function with n_prev=%d", n_prev);

    Tensor* a = prev[0];
    Tensor* b = prev[1];

    int size = numel(a->shape, a->ndim);
    float dout = out->grad->data->data[0];

    if (!is_contiguous(a) || !is_contiguous(b))
    {
        if (a->requires_grad)
        {
            for (int linear = 0; linear < size; ++linear)
            {
                int idx = linear;
                int a_offset = 0, b_offset = 0;

                for (int d = a->ndim - 1; d >= 0; --d)
                {
                    int coord = idx % a->shape[d];
                    idx /= a->shape[d];

                    a_offset += coord * a->strides[d];
                    b_offset += coord * b->strides[d];
                }
                a->grad->data->data[a_offset] += dout * b->data->data[b_offset];
            }
        }

        if (b->requires_grad)
        {
            for (int linear = 0; linear < size; ++linear)
            {
                int idx = linear;
                int a_offset = 0, b_offset = 0;

                for (int d = a->ndim - 1; d >= 0; --d)
                {
                    int coord = idx % a->shape[d];
                    idx /= a->shape[d];

                    a_offset += coord * a->strides[d];
                    b_offset += coord * b->strides[d];
                }
                b->grad->data->data[b_offset] += dout * a->data->data[a_offset];
            }
        }
    }
    else
    {
        if (a->requires_grad)
        {
            int i = 0;
            __m256 dout_vec = _mm256_set1_ps(dout);
            for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
            {
                __m256 a_grad = _mm256_loadu_ps(a->grad->data->data + i);
                __m256 b_data = _mm256_loadu_ps(b->data->data + i);
                __m256 da = _mm256_fmadd_ps(dout_vec, b_data, a_grad);
                _mm256_storeu_ps(a->grad->data->data + i, da);
            }

            for (; i < size; ++i)
            {
                a->grad->data->data[i] += dout * b->data->data[i];
            }
        }

        if (b->requires_grad)
        {
            int i = 0;
            __m256 dout_vec = _mm256_set1_ps(dout);
            for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
            {
                __m256 b_grad = _mm256_loadu_ps(b->grad->data->data + i);
                __m256 a_data = _mm256_loadu_ps(a->data->data + i);
                __m256 db = _mm256_fmadd_ps(dout_vec, a_data, b_grad);
                _mm256_storeu_ps(b->grad->data->data + i, db);
            }

            for (; i < size; ++i)
            {
                b->grad->data->data[i] += dout * a->data->data[i];
            }
        }
    }

    LOG_INFO("dot_grad_op_cpu: Exiting function.");
}
