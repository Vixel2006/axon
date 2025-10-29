#include "autograd/cpu/binary/common.h"

void mul_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("mul_grad_op_cpu: Entering function with n_prev=%d", n_prev);

    // Error checking for null tensors
    if (!out || !out->grad->data->data || !prev)
    {
        LOG_ERROR("mul_grad_op: Output tensor, output gradient, or previous tensors array is NULL! "
                  "out=%p, out->grad=%p, prev=%p",
                  (void*) out, (void*) out->grad->data->data, (void*) prev);
        assert(0 &&
               "mul_grad_op: Output tensor, output gradient, or previous tensors array is NULL!");
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
            binary_grad_noncontig(out, a, b, binary_mul_da, binary_mul_db);
        }
        else
        {
            if (a->requires_grad)
            {
                int i = 0;
                for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
                {
                    __m256 a_grad = _mm256_loadu_ps(a->grad->data->data + i);
                    __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
                    __m256 b_data = _mm256_loadu_ps(b->data->data + i);
                    __m256 da = _mm256_fmadd_ps(dout, b_data, a_grad);
                    _mm256_storeu_ps(a->grad->data->data + i, da);
                }

                for (; i < size; ++i)
                {
                    a->grad->data->data[i] += out->grad->data->data[i] * b->data->data[i];
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
                    __m256 db = _mm256_fmadd_ps(dout, a_data, b_grad);
                    _mm256_storeu_ps(b->grad->data->data + i, db);
                }

                for (; i < size; ++i)
                {
                    b->grad->data->data[i] += out->grad->data->data[i] * a->data->data[i];
                }
            }
        }
    }
    else if (n_prev == 1 && extras != NULL)
    {
        Tensor* a = prev[0];
        float b_scalar = *((float*) extras);

        if (!is_contiguous(a) || !is_contiguous(out) ||
            !shapes_equal(a->shape, a->ndim, out->shape, out->ndim))
        {
            unary_grad_noncontig(out, a, b_scalar, unary_mul_da);
        }
        else
        {
            if (a->requires_grad)
            {
                __m256 scalar_b = _mm256_set1_ps(b_scalar);
                int i = 0;
                for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
                {
                    __m256 a_grad = _mm256_loadu_ps(a->grad->data->data + i);
                    __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
                    __m256 da = _mm256_fmadd_ps(scalar_b, dout, a_grad);
                    _mm256_storeu_ps(a->grad->data->data + i, da);
                }

                for (; i < size; ++i)
                {
                    a->grad->data->data[i] += out->grad->data->data[i] * b_scalar;
                }
            }
        }
    }
    LOG_INFO("mul_grad_op_cpu: Exiting function.");
}
