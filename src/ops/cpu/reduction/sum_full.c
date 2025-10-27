#include "ops/cpu/reduction.h"

void sum_full_op_cpu(Tensor* a, Tensor* out)
{
    LOG_INFO("sum_full_op_cpu: Entering function");

    size_t total_elements = numel(a->shape, a->ndim);

    float* total_sum_ptr = (float*) malloc(sizeof(float));
    if (!total_sum_ptr)
    {
        LOG_ERROR("sum_full_op: Failed to allocate memory for result");
        return;
    }
    total_sum_ptr[0] = 0.0f;

    if (total_elements == 0)
    {
        from_data(out, total_sum_ptr);
        return;
    }

    bool is_a_contiguous = is_contiguous(a);

    if (is_a_contiguous)
    {
        __m256 sum_vec = _mm256_setzero_ps();
        size_t i = 0;

        for (; i + SIMD_WIDTH - 1 < total_elements; i += SIMD_WIDTH)
        {
            __m256 vec_a = _mm256_loadu_ps(a->data->data + i);
            sum_vec = _mm256_add_ps(sum_vec, vec_a);
        }

        // Horizontal sum
        sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
        sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
        __m128 lo_half = _mm256_extractf128_ps(sum_vec, 0);
        __m128 hi_half = _mm256_extractf128_ps(sum_vec, 1);
        __m128 total_sum_m128 = _mm_add_ps(lo_half, hi_half);
        total_sum_ptr[0] = _mm_cvtss_f32(total_sum_m128);

        for (; i < total_elements; ++i)
        {
            total_sum_ptr[0] += a->data->data[i];
        }
    }
    else
    {
        int* coords = (int*) malloc(a->ndim * sizeof(int));
        if (!coords)
        {
            LOG_ERROR("sum_full_op: Failed to allocate memory for coords.");
            SAFE_FREE(&total_sum_ptr, free);
            return;
        }

        memset(coords, 0, a->ndim * sizeof(int));

        for (size_t elem = 0; elem < total_elements; ++elem)
        {
            size_t flat_idx = 0;
            for (int d = 0; d < a->ndim; ++d)
            {
                flat_idx += (size_t) coords[d] * a->strides[d];
            }

            total_sum_ptr[0] += a->data->data[flat_idx];

            int carry = 1;
            for (int d = a->ndim - 1; d >= 0 && carry; --d)
            {
                coords[d] += carry;
                if (coords[d] < a->shape[d])
                {
                    carry = 0;
                }
                else
                {
                    coords[d] = 0;
                    carry = 1;
                }
            }
        }
        SAFE_FREE(&coords, free);
    }

    from_data(out, total_sum_ptr);
    SAFE_FREE(&total_sum_ptr, free);
}
