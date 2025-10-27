#include "ops/cpu/init.h" // For from_data
#include "ops/cpu/unary.h"

void exp_op_cpu(Tensor* in, Tensor* out)
{
    LOG_INFO("exp_op_cpu: Entering function");

    if (!check_tensors_unary(in, out, "exp_op")) return;

    int size = numel(in->shape, in->ndim);
    float* data = alloc_tensor_data(size, "exp_op");
    if (!data) return;

    if (!can_use_simd_unary(in, out))
    {
        int offset_in, offset_out;
        for (int linear = 0; linear < size; ++linear)
        {
            COMPUTE_UNARY_OFFSETS(linear, in, offset_in, offset_out);
            data[offset_out] = expf(in->data->data[offset_in]);
        }
    }
    else
    {
        int i = 0;

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
        {
            __m256 x = _mm256_loadu_ps(in->data->data + i);
            __m256 z = Sleef_expf8_u10avx2(x);
            _mm256_storeu_ps(data + i, z);
        }

        for (; i < size; ++i)
        {
            data[i] = expf(in->data->data[i]);
        }
    }
    from_data(out, data);
    SAFE_FREE(&data, free);
}
