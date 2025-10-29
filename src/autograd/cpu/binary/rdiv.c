#include "autograd/cpu/binary/common.h"

void rdiv_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("rdiv_grad_op_cpu: Entering function with n_prev=%d", n_prev);

    Tensor* a = prev[0];
    float b = *((float*) extras);

    int size = numel(out->shape, out->ndim);

    if (a->requires_grad)
    {
        if (!is_contiguous(a) || !is_contiguous(out) ||
            !shapes_equal(a->shape, a->ndim, out->shape, out->ndim))
        {
            unary_grad_noncontig(out, a, b, unary_rdiv_da);
        }
        else
        {
            int i = 0;
            __m256 neg_b = _mm256_set1_ps(-b);
            for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
            {
                __m256 a_grad = _mm256_loadu_ps(a->grad->data->data + i);
                __m256 a_data = _mm256_loadu_ps(a->data->data + i);
                __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
                __m256 a_squared = _mm256_mul_ps(a_data, a_data);
                __m256 da = _mm256_fmadd_ps(_mm256_div_ps(neg_b, a_squared), dout, a_grad);
                _mm256_storeu_ps(a->grad->data->data + i, da);
            }

            for (; i < size; ++i)
            {
                float aa = a->data->data[i];
                a->grad->data->data[i] +=
                    (aa != 0.0f ? out->grad->data->data[i] * (-b) / (aa * aa) : 0.0f);
            }
        }
    }
    LOG_INFO("rdiv_grad_op_cpu: Exiting function.");
}
