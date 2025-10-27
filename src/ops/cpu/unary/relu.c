#include "ops/cpu/init.h" // For from_data
#include "ops/cpu/unary.h"

void relu_op_cpu(Tensor* in, Tensor* out)
{
    LOG_INFO("relu_op_cpu: Entering function");

    if (!check_tensors_unary(in, out, "relu_op")) return;

    int size = numel(in->shape, in->ndim);
    float* data = alloc_tensor_data(size, "relu_op");
    if (!data) return;

    if (!can_use_simd_unary(in, out))
    {
        int offset_in, offset_out;
        for (int linear = 0; linear < size; ++linear)
        {
            COMPUTE_UNARY_OFFSETS(linear, in, offset_in, offset_out);
            data[offset_out] = in->data->data[offset_in] > 0 ? in->data->data[offset_in] : 0.0f;
        }
    }
    else
    {
        int i = 0;
        __m256 zeros = _mm256_setzero_ps();

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
        {
            __m256 vin = _mm256_loadu_ps(in->data->data + i);
            __m256 vout = _mm256_max_ps(vin, zeros);
            _mm256_storeu_ps(data + i, vout);
        }

        for (; i < size; ++i)
        {
            data[i] = in->data->data[i] > 0.0f ? in->data->data[i] : 0.0f;
        }
    }
    from_data(out, data);
    SAFE_FREE(&data, free);
}
