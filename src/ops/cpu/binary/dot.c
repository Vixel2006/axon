#include "ops/cpu/binary.h"

void dot_op_cpu(Tensor* a, Tensor* b, Tensor* out)
{
    if (!check_tensors(a, b, out, "dot_op")) return;

    int size = numel(a->shape, a->ndim);
    float* data = alloc_tensor_data(1, "dot_op");
    if (!data) return;

    if (!can_use_simd(a, b, out))
    {
        float sum = 0.0f;
        int a_offset, b_offset, dummy_out_offset;
        for (int linear = 0; linear < size; ++linear)
        {
            COMPUTE_OFFSETS(linear, a, a->strides, b->strides, a->strides, a_offset, b_offset,
                            dummy_out_offset);
            sum += a->data->data[a_offset] * b->data->data[b_offset];
        }
        data[0] = sum;
    }
    else
    {
        float sum = 0.0f;
        int i = 0;

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
        {
            __m256 x = _mm256_loadu_ps(a->data->data + i);
            __m256 y = _mm256_loadu_ps(b->data->data + i);
            __m256 prod_vec = _mm256_mul_ps(x, y);

            __m128 sum_high = _mm256_extractf128_ps(prod_vec, 1);
            __m128 sum_low = _mm256_castps256_ps128(prod_vec);
            __m128 sum128 = _mm_add_ps(sum_high, sum_low);
            __m128 shuf = _mm_movehdup_ps(sum128);
            __m128 sums = _mm_add_ps(sum128, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            sum += _mm_cvtss_f32(sums);
        }

        for (; i < size; ++i)
        {
            sum += a->data->data[i] * b->data->data[i];
        }
        data[0] = sum;
    }
    from_data(out, data);
    SAFE_FREE(&data, free);
}
