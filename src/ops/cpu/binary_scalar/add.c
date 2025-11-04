#include "ops/cpu/binary_scalar.h"
#include "ops/cpu/init.h"

#define SIMD_WIDTH 8

void add_scalar_op_cpu(Tensor* a, float b, Tensor* out)
{
    LOG_INFO("add_scalar_op: Performing scalar addition (scalar=%.2f)", b);

    int size = numel(a->shape, a->ndim);
    float* data = malloc(sizeof(float) * size);

    if (!is_contiguous(a) || !is_contiguous(out))
    {
        for (int idx = 0; idx < size; ++idx)
        {
            int offset_a = 0;
            int offset_out = 0;
            int tmp = idx;

            for (int d = a->ndim - 1; d >= 0; --d)
            {
                int coord = tmp % a->shape[d];
                tmp /= a->shape[d];
                offset_a += coord * a->strides[d];
                offset_out += coord * out->strides[d];
            }

            data[offset_out] = a->data->data[offset_a] + b;
        }
    }
    else
    {
        int i = 0;
        __m256 scalar = _mm256_set1_ps(b);

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
        {
            __m256 x = _mm256_loadu_ps(a->data->data + i);
            __m256 z = _mm256_add_ps(x, scalar);
            _mm256_storeu_ps(data + i, z);
        }

        for (; i < size; ++i)
        {
            data[i] = a->data->data[i] + b;
        }
    }
    from_data(out, data);
    SAFE_FREE(&data, free);
}
