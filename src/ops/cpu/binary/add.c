#include "ops/cpu/binary.h"
#include "ops/cpu/init.h"

void add_op_cpu(Tensor* a, Tensor* b, Tensor* out)
{
    LOG_INFO("add_op_cpu: Entering function");

    if (!check_tensors(a, b, out, "add_op")) return;

    int size = numel(out->shape, out->ndim);
    float* data = alloc_tensor_data(size, "add_op");
    if (!data) return;

    if (!can_use_simd(a, b, out))
    {
        int a_offset, b_offset, out_offset;
        for (int linear = 0; linear < size; ++linear)
        {
            COMPUTE_OFFSETS(linear, out, a->strides, b->strides, out->strides, a_offset, b_offset,
                            out_offset);
            data[out_offset] = a->data->data[a_offset] + b->data->data[b_offset];
        }
    }
    else
    {
        int i = 0;
        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
        {
            __m256 x = _mm256_loadu_ps(a->data->data + i);
            __m256 y = _mm256_loadu_ps(b->data->data + i);
            __m256 z = _mm256_add_ps(x, y);
            _mm256_storeu_ps(data + i, z);
        }
        for (; i < size; ++i)
        {
            data[i] = a->data->data[i] + b->data->data[i];
        }
    }
    from_data(out, data);
    SAFE_FREE(&data, free);
}
