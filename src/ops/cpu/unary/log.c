#include "ops/cpu/init.h" // For from_data
#include "ops/cpu/unary.h"

void log_op_cpu(Tensor* in, Tensor* out)
{
    LOG_INFO("log_op_cpu: Entering function");

    if (!check_tensors_unary(in, out, "log_op")) return;

    int size = numel(in->shape, in->ndim);

    bool is_contig = is_contiguous(in);
    for (int linear = 0; linear < size; ++linear)
    {
        int offset_in;
        if (is_contig)
        {
            offset_in = linear;
        }
        else
        {
            int idx = linear;
            offset_in = 0;
            for (int d = in->ndim - 1; d >= 0; --d)
            {
                int coord = idx % in->shape[d];
                idx /= in->shape[d];
                offset_in += coord * in->strides[d];
            }
        }
        float val = in->data->data[offset_in];
        if (val < 0.0f)
        {
            LOG_ERROR("log_op ERROR: Input value at linear index %d is negative (%.4f)! "
                      "Logarithm of negative number is undefined.",
                      linear, val);
        }
        else if (val == 0.0f)
        {
            LOG_WARN("log_op WARNING: Input value at linear index %d is zero. "
                     "Logarithm of zero is -INF.",
                     linear);
        }
    }

    float* data = alloc_tensor_data(size, "log_op");
    if (!data) return;

    if (!can_use_simd_unary(in, out))
    {
        int offset_in, offset_out;
        for (int linear = 0; linear < size; ++linear)
        {
            COMPUTE_UNARY_OFFSETS(linear, in, offset_in, offset_out);
            data[offset_out] = SAFE_LOGF(in->data->data[offset_in]);
        }
    }
    else
    {
        int i = 0;

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
        {
            __m256 x = _mm256_loadu_ps(in->data->data + i);
            __m256 minimum = _mm256_set1_ps(EPS);
            __m256 clamped = _mm256_max_ps(x, minimum);
            __m256 z = Sleef_logf8_u10avx2(clamped);
            _mm256_storeu_ps(data + i, z);
        }

        for (; i < size; ++i)
        {
            data[i] = SAFE_LOGF(in->data->data[i]);
        }
    }
    from_data(out, data);
    SAFE_FREE(&data, free);
}
