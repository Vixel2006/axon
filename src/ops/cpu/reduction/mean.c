#include "ops/cpu/reduction.h"

void mean_op_cpu(Tensor* a, Tensor* out, int axis, bool keepdim)
{
    LOG_INFO("mean_op_cpu: Entering function with axis=%d, keepdim=%d", axis, keepdim);

    if (axis < 0 || axis >= a->ndim)
    {
        LOG_ERROR("mean_op: Invalid axis %d for tensor with %d dimensions", axis, a->ndim);
        return;
    }

    // First compute the sum. out->data will be set by sum_op.
    sum_op_cpu(a, out, axis, keepdim);

    // Check if sum_op failed or returned empty data
    if (!out->data || !out->data->data)
    {
        LOG_ERROR("mean_op: sum_op failed to produce valid output data.");
        return;
    }

    // Then divide by the reduction size
    float reduction_size = (float) a->shape[axis];
    size_t size = numel(out->shape, out->ndim);

    // Operate directly on out->data->data
    if (reduction_size > 0)
    {
        // SIMD path for contiguous data
        if (is_contiguous(out))
        {
            __m256 div_vec = _mm256_set1_ps(reduction_size);
            size_t i = 0;
            for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
            {
                __m256 vec_out = _mm256_loadu_ps(out->data->data + i);
                vec_out = _mm256_div_ps(vec_out, div_vec);
                _mm256_storeu_ps(out->data->data + i, vec_out);
            }
            for (; i < size; i++)
            {
                out->data->data[i] /= reduction_size;
            }
        }
        else
        {
            // Scalar path for non-contiguous data
            // This case needs to be careful with strides if out is not contiguous
            // For now, assuming out->data->data is a flat array, and we modify it directly.
            // If out is non-contiguous, it means data access needs to use strides.
            // However, from_data typically makes 'out' contiguous or handles data pointers.
            // For safety, let's just do scalar division on the underlying data buffer.
            for (size_t i = 0; i < size; i++)
            {
                out->data->data[i] /= reduction_size;
            }
        }
    }
    // If reduction_size is 0, elements should probably remain 0 or NaN, current code keeps them as
    // 0.
}
