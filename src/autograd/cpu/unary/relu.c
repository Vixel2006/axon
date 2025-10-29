#include "autograd/cpu/unary/common.h"

void relu_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("relu_grad_op_cpu: Entering function with n_prev=%d", n_prev);

    Tensor* a = prev[0];

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
                if (a->data->data[a_offset] > 0)
                {
                    a->grad->data->data[a_offset] += out->grad->data->data[out_offset];
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
            __m256 mask = _mm256_cmp_ps(va, _mm256_setzero_ps(), _CMP_GT_OQ);
            __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
            __m256 dmasked = _mm256_and_ps(dout, mask);
            __m256 dprev = _mm256_loadu_ps(a->grad->data->data + i);
            dprev = _mm256_add_ps(dprev, dmasked);
            _mm256_storeu_ps(a->grad->data->data + i, dprev);
        }

        for (; i < size; ++i)
        {
            if (a->data->data[i] > 0)
            {
                a->grad->data->data[i] += out->grad->data->data[i];
            }
        }
    }
    LOG_INFO("relu_grad_op_cpu: Exiting function.");
}
