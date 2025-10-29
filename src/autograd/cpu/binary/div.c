#include "autograd/cpu/binary/common.h"

void div_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("div_grad_op_cpu: Entering function with n_prev=%d", n_prev);

    // Basic null pointer checks
    if (!out || !prev)
    {
        LOG_ERROR("div_grad_op: Output tensor or previous tensors array is NULL! out=%p, prev=%p",
                  (void*) out, (void*) prev);
        assert(0 && "div_grad_op: Output tensor or previous tensors array is NULL!");
    }

    if (!out->grad->data->data)
    {
        LOG_ERROR("div_grad_op: Output gradient is NULL! out->grad=%p",
                  (void*) out->grad->data->data);
        assert(0 && "div_grad_op: Output gradient is NULL!");
    }

    if (n_prev != 2 && n_prev != 1)
    {
        LOG_ERROR("div_grad_op: Invalid number of previous tensors: %d. Expected 1 or 2.", n_prev);
        assert(0 && "div_grad_op: Invalid number of previous tensors. Expected 1 or 2.");
    }

    if (n_prev == 2)
    {
        if (!prev[0] || !prev[1])
        {
            LOG_ERROR("div_grad_op: One or both previous tensors are NULL! prev[0]=%p, prev[1]=%p",
                      (void*) prev[0], (void*) prev[1]);
            assert(0 && "div_grad_op: One or both previous tensors are NULL!");
        }
        if (!prev[0]->data->data || !prev[1]->data->data)
        {
            LOG_ERROR("div_grad_op: One or both previous tensors' data are NULL!");
            assert(0 && "div_grad_op: One or both previous tensors' data are NULL!");
        }
        if (prev[0]->requires_grad && prev[0]->grad->data->data == NULL)
        {
            LOG_ERROR("div_grad_op: Previous tensor 0 requires grad but its grad is NULL!");
            assert(0 && "div_grad_op: Previous tensor 0 requires grad but its grad is NULL!");
        }
        if (prev[1]->requires_grad && prev[1]->grad->data->data == NULL)
        {
            LOG_ERROR("div_grad_op: Previous tensor 1 requires grad but its grad is NULL!");
            assert(0 && "div_grad_op: Previous tensor 1 requires grad but its grad is NULL!");
        }
    }

    int size = numel(out->shape, out->ndim);

    if (n_prev == 2)
    {
        Tensor* a = prev[0];
        Tensor* b = prev[1];

        if (!is_contiguous(a) || !is_contiguous(b) || !is_contiguous(out) ||
            !shapes_equal(a->shape, a->ndim, out->shape, out->ndim) ||
            !shapes_equal(b->shape, b->ndim, out->shape, out->ndim))
        {
            binary_grad_noncontig(out, a, b, binary_div_da, binary_div_db);
        }
        else
        {
            __m256 zero = _mm256_setzero_ps();
            if (a->requires_grad)
            {
                int i = 0;
                for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
                {
                    __m256 a_grad = _mm256_loadu_ps(a->grad->data->data + i);
                    __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
                    __m256 b_data = _mm256_loadu_ps(b->data->data + i);
                    __m256 mask = _mm256_cmp_ps(b_data, zero, _CMP_EQ_OQ);
                    __m256 div = _mm256_div_ps(dout, b_data);
                    div = _mm256_blendv_ps(div, zero, mask);
                    __m256 da = _mm256_add_ps(a_grad, div);
                    _mm256_storeu_ps(a->grad->data->data + i, da);
                }
                for (; i < size; ++i)
                {
                    float bb = b->data->data[i];
                    a->grad->data->data[i] += (bb != 0.0f ? out->grad->data->data[i] / bb : 0.0f);
                }
            }

            if (b->requires_grad)
            {
                int i = 0;
                for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
                {
                    __m256 b_grad = _mm256_loadu_ps(b->grad->data->data + i);
                    __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
                    __m256 a_data = _mm256_loadu_ps(a->data->data + i);
                    __m256 b_data = _mm256_loadu_ps(b->data->data + i);
                    __m256 b_sq = _mm256_mul_ps(b_data, b_data);
                    __m256 mask = _mm256_cmp_ps(b_sq, zero, _CMP_EQ_OQ);
                    __m256 term = _mm256_mul_ps(dout, a_data);
                    __m256 div = _mm256_div_ps(term, b_sq);
                    div = _mm256_blendv_ps(div, zero, mask);
                    __m256 db = _mm256_sub_ps(b_grad, div);
                    _mm256_storeu_ps(b->grad->data->data + i, db);
                }
                for (; i < size; ++i)
                {
                    float bb = b->data->data[i];
                    if (bb != 0.0f)
                    {
                        b->grad->data->data[i] -=
                            out->grad->data->data[i] * a->data->data[i] / (bb * bb);
                    }
                }
            }
        }
    }
    else if (n_prev == 1 && extras != NULL)
    {
        Tensor* a = prev[0];
        float b = *((float*) extras);

        if (!is_contiguous(a) || !is_contiguous(out) ||
            !shapes_equal(a->shape, a->ndim, out->shape, out->ndim))
        {
            unary_grad_noncontig(out, a, b, unary_div_da);
        }
        else
        {
            if (a->requires_grad)
            {
                float inv_b = (b != 0.0f ? 1.0f / b : 0.0f);
                __m256 inv = _mm256_set1_ps(inv_b);
                int i = 0;
                for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
                {
                    __m256 a_grad = _mm256_loadu_ps(a->grad->data->data + i);
                    __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
                    __m256 da = _mm256_fmadd_ps(inv, dout, a_grad);
                    _mm256_storeu_ps(a->grad->data->data + i, da);
                }

                for (; i < size; ++i)
                {
                    a->grad->data->data[i] += (b != 0.0f ? out->grad->data->data[i] / b : 0.0f);
                }
            }
        }
    }
    LOG_INFO("div_grad_op_cpu: Exiting function.");
}
