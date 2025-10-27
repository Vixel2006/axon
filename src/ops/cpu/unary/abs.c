#include "ops/cpu/init.h" // For from_data
#include "ops/cpu/unary.h"

void abs_op_cpu(Tensor* in, Tensor* out)
{
    LOG_INFO("abs_op_cpu: Entering function");

    if (!check_tensors_unary(in, out, "abs_op")) return;

    int size = numel(in->shape, in->ndim);
    float* data = alloc_tensor_data(size, "abs_op");
    if (!data) return;

    if (!can_use_simd_unary(in, out))
    {
        int offset_in, offset_out;
        for (int linear = 0; linear < size; ++linear)
        {
            COMPUTE_UNARY_OFFSETS(linear, in, offset_in, offset_out);
            data[offset_out] = in->data->data[offset_in] >= 0.0f ? in->data->data[offset_in]
                                                                 : 0.0f - in->data->data[offset_in];
        }
    }
    else
    {
        int i = 0;
        __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)); // mask to remove sign bit

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
        {
            __m256 x = _mm256_loadu_ps(in->data->data + i);
            __m256 y = _mm256_and_ps(x, mask);
            _mm256_storeu_ps(data + i, y);
        }

        for (; i < size; ++i)
        {
            data[i] = in->data->data[i] >= 0 ? in->data->data[i] : 0.0f - in->data->data[i];
        }
    }
    from_data(out, data);
    SAFE_FREE(&data, free);
}
