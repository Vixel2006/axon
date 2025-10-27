#include "ops/cpu/init.h" // For from_data
#include "ops/cpu/unary.h"

void clip_op_cpu(Tensor* in, Tensor* out, float min_val, float max_val)
{
    LOG_INFO("clip_op_cpu: Entering function with min_val=%.2f, max_val=%.2f", min_val, max_val);

    if (!check_tensors_unary(in, out, "clip_op")) return;

    int size = numel(in->shape, in->ndim);
    float* data = alloc_tensor_data(size, "clip_op");
    if (!data) return;

    if (!can_use_simd_unary(in, out))
    {
        int offset_in, offset_out;
        for (int linear = 0; linear < size; ++linear)
        {
            COMPUTE_UNARY_OFFSETS(linear, in, offset_in, offset_out);
            float val = in->data->data[offset_in];
            if (val < min_val)
            {
                data[offset_out] = min_val;
            }
            else if (val > max_val)
            {
                data[offset_out] = max_val;
            }
            else
            {
                data[offset_out] = val;
            }
        }
    }
    else
    {
        int i = 0;
        __m256 v_min = _mm256_set1_ps(min_val);
        __m256 v_max = _mm256_set1_ps(max_val);

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
        {
            __m256 x = _mm256_loadu_ps(in->data->data + i);
            __m256 y = _mm256_max_ps(x, v_min);
            y = _mm256_min_ps(y, v_max);
            _mm256_storeu_ps(data + i, y);
        }

        for (; i < size; ++i)
        {
            float val = in->data->data[i];
            if (val < min_val)
            {
                data[i] = min_val;
            }
            else if (val > max_val)
            {
                data[i] = max_val;
            }
            else
            {
                data[i] = val;
            }
        }
    }
    from_data(out, data);
    SAFE_FREE(&data, free);
}
