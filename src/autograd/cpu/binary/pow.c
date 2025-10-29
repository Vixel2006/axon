#include "autograd/cpu/binary/common.h"

void pow_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("pow_grad_op_cpu: Entering function with n_prev=%d", n_prev);

    // Error checking for null tensors
    if (!out || !out->grad->data->data || !prev)
    {
        LOG_ERROR("pow_grad_op: Output tensor, output gradient, or previous tensors array is NULL! "
                  "out=%p, out->grad=%p, prev=%p",
                  (void*) out, (void*) out->grad->data->data, (void*) prev);
        assert(0 &&
               "pow_grad_op: Output tensor, output gradient, or previous tensors array is NULL!");
    }

    if (n_prev != 1 && n_prev != 2)
    {
        LOG_ERROR("pow_grad_op: Invalid number of previous tensors: %d. Expected 1 or 2.", n_prev);
        assert(0 && "pow_grad_op: Invalid number of previous tensors. Expected 1 or 2.");
    }

    Tensor* a = prev[0];
    int size = numel(out->shape, out->ndim);

    if (n_prev == 1)
    { // Tensor ** scalar
        float b_scalar = *((float*) extras);

        if (!a->requires_grad) return;

        if (!is_contiguous(a) || !is_contiguous(out) ||
            !shapes_equal(a->shape, a->ndim, out->shape, out->ndim))
        {
            unary_grad_noncontig(out, a, b_scalar, unary_pow_da);
        }
        else
        {
            int i = 0;
            __m256 scalar_b = _mm256_set1_ps(b_scalar);
            float c = b_scalar - 1.0f;
            __m256 scalar_bm1 = _mm256_set1_ps(c);
            __m256 zero = _mm256_setzero_ps();

            for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
            {
                __m256 x = _mm256_loadu_ps(a->data->data + i);
                __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
                __m256 agrad = _mm256_loadu_ps(a->grad->data->data + i);

                __m256 x_pow = Sleef_powf8_u10avx2(x, scalar_bm1);
                __m256 coeff = _mm256_mul_ps(scalar_b, x_pow);

                // mask problematic case: x==0 && (b-1)<0
                __m256 zero_mask = _mm256_cmp_ps(x, zero, _CMP_EQ_OQ);
                __m256 neg_exp_mask = _mm256_cmp_ps(scalar_bm1, zero, _CMP_LT_OQ);
                __m256 problem_mask = _mm256_and_ps(zero_mask, neg_exp_mask);
                coeff = _mm256_blendv_ps(coeff, zero, problem_mask);

                __m256 da = _mm256_fmadd_ps(dout, coeff, agrad);
                _mm256_storeu_ps(a->grad->data->data + i, da);
            }

            for (; i < size; ++i)
            {
                float x = a->data->data[i];
                float grad_val = 0.0f;

                if (!(x == 0.0f && (b_scalar - 1.0f) < 0.0f))
                {
                    grad_val = b_scalar * powf(x, b_scalar - 1.0f);
                }
                a->grad->data->data[i] += out->grad->data->data[i] * grad_val;
            }
        }
    }
    else if (n_prev == 2)
    { // Tensor ** Tensor
        Tensor* b_tensor = prev[1];

        if (!a->requires_grad && !b_tensor->requires_grad) return;

        if (!is_contiguous(a) || !is_contiguous(b_tensor) || !is_contiguous(out) ||
            !shapes_equal(a->shape, a->ndim, out->shape, out->ndim) ||
            !shapes_equal(b_tensor->shape, b_tensor->ndim, out->shape, out->ndim))
        {
            binary_grad_noncontig(out, a, b_tensor, binary_pow_da, binary_pow_db);
        }
        else
        {
            // Contiguous case with SIMD (AVX2)
            __m256 zero = _mm256_setzero_ps();
            __m256 one = _mm256_set1_ps(1.0f);

            int i = 0; // Declare i here
            for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
            {
                __m256 x = _mm256_loadu_ps(a->data->data + i);
                __m256 y = _mm256_loadu_ps(b_tensor->data->data + i);
                __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);

                // Gradient for 'a': dout * y * x^(y-1)
                if (a->requires_grad)
                {
                    __m256 agrad = _mm256_loadu_ps(a->grad->data->data + i);
                    __m256 y_minus_one = _mm256_sub_ps(y, one);
                    __m256 x_pow_y_minus_one = Sleef_powf8_u10avx2(x, y_minus_one);
                    __m256 coeff_a = _mm256_mul_ps(y, x_pow_y_minus_one);

                    // Mask problematic case: x==0 && (y-1)<0
                    __m256 zero_mask_x = _mm256_cmp_ps(x, zero, _CMP_EQ_OQ);
                    __m256 neg_exp_mask_y = _mm256_cmp_ps(y_minus_one, zero, _CMP_LT_OQ);
                    __m256 problem_mask_a = _mm256_and_ps(zero_mask_x, neg_exp_mask_y);
                    coeff_a = _mm256_blendv_ps(coeff_a, zero, problem_mask_a);

                    __m256 da = _mm256_fmadd_ps(dout, coeff_a, agrad);
                    _mm256_storeu_ps(a->grad->data->data + i, da);
                }

                // Gradient for 'b': dout * x^y * log(x)
                if (b_tensor->requires_grad)
                {
                    __m256 bgrad = _mm256_loadu_ps(b_tensor->grad->data->data + i);
                    __m256 x_pow_y = Sleef_powf8_u10avx2(x, y);
                    __m256 log_x = Sleef_logf8_u10avx2(x);

                    // Mask for x <= 0 for log(x)
                    __m256 non_positive_x_mask = _mm256_cmp_ps(x, zero, _CMP_LE_OQ);
                    log_x = _mm256_blendv_ps(log_x, zero,
                                             non_positive_x_mask); // Set log(x) to 0 if x <= 0

                    __m256 coeff_b = _mm256_mul_ps(x_pow_y, log_x);
                    __m256 db = _mm256_fmadd_ps(dout, coeff_b, bgrad);
                    _mm256_storeu_ps(b_tensor->grad->data->data + i, db);
                }
            }

            // Handle remaining elements (non-SIMD)
            for (; i < size; ++i)
            {
                float x = a->data->data[i];
                float y = b_tensor->data->data[i];
                float dout = out->grad->data->data[i];

                // Gradient for 'a'
                if (a->requires_grad)
                {
                    float grad_a_val = 0.0f;
                    if (!(x == 0.0f && (y - 1.0f) < 0.0f))
                    {
                        grad_a_val = y * powf(x, y - 1.0f);
                    }
                    a->grad->data->data[i] += dout * grad_a_val;
                }

                // Gradient for 'b'
                if (b_tensor->requires_grad)
                {
                    float grad_b_val = 0.0f;
                    if (x > 0.0f)
                    {
                        grad_b_val = powf(x, y) * logf(x);
                    }
                    b_tensor->grad->data->data[i] += dout * grad_b_val;
                }
            }
        }
    }
    LOG_INFO("pow_grad_op_cpu: Exiting function.");
}
