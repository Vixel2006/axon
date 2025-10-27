#include "ops/cpu/init.h" // For from_data
#include "ops/cpu/unary.h"

void neg_op_cpu(Tensor* in, Tensor* out)
{
    LOG_INFO("neg_op_cpu: Entering function");

    if (!check_tensors_unary(in, out, "neg_op")) return;

    int size = numel(in->shape, in->ndim);
    float* data = alloc_tensor_data(size, "neg_op");
    if (!data) return;

    if (!can_use_simd_unary(in, out))
    {
        int offset_in, offset_out;
        for (int linear = 0; linear < size; ++linear)
        {
            COMPUTE_UNARY_OFFSETS(linear, in, offset_in, offset_out);
            data[offset_out] = 0.0f - in->data->data[offset_in];
        }
    }
    else
    {
        int i = 0;

        __m256 zeros = _mm256_setzero_ps();

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
        {
            __m256 x = _mm256_loadu_ps(in->data->data + i);
            __m256 y = _mm256_sub_ps(zeros, x);
            _mm256_storeu_ps(data + i, y);
        }

        for (; i < size; ++i)
        {
            data[i] = 0.0f - in->data->data[i];
        }
    }
    from_data(out, data);
    SAFE_FREE(&data, free);
}
