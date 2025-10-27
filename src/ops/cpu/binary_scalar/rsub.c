#include "ops/cpu/binary_scalar.h"

#define SIMD_WIDTH 8

void rsub_scalar_op_cpu(Tensor* a, float b, Tensor* out)
{
    LOG_INFO("rsub_scalar_op: Performing reverse scalar subtraction (scalar=%.2f)", b);

    // Error checking for null tensors
    if (!a || !out)
    {
        LOG_ERROR("rsub_scalar_op ERROR: Input or output tensor is NULL! a=%p, out=%p", (void*) a,
                  (void*) out);
        return;
    }

    int size = numel(a->shape, a->ndim);
    float* data = malloc(sizeof(float) * size);

    if (!is_contiguous(a) || !is_contiguous(out))
    {
        for (int idx = 0; idx < size; ++idx)
        {
            int a_offset = 0;
            int out_offset = 0;
            int tmp = idx;

            for (int d = a->ndim - 1; d >= 0; --d)
            {
                int coord = tmp % a->shape[d];
                tmp /= a->shape[d];

                a_offset += coord * a->strides[d];
                out_offset += coord * out->strides[d];
            }

            data[out_offset] = b - a->data->data[a_offset];
        }
    }
    else
    {
        int i = 0;
        __m256 scalar = _mm256_set1_ps(b);

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
        {
            __m256 x = _mm256_loadu_ps(a->data->data + i);
            __m256 z = _mm256_sub_ps(scalar, x);
            _mm256_storeu_ps(data + i, z);
        }

        for (; i < size; ++i)
        {
            data[i] = b - a->data->data[i];
        }
    }
    from_data(out, data);
    SAFE_FREE(&data, free);
}
