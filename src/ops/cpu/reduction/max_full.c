#include "ops/cpu/reduction.h"

void max_full_op_cpu(Tensor* a, Tensor* out)
{
    LOG_INFO("max_full_op_cpu: Entering function");

    size_t total_elements = numel(a->shape, a->ndim);

    float* data = (float*) malloc(sizeof(float)); // Allocate for a single float (scalar result)
    if (!data)
    {
        LOG_ERROR("max_full_op: Failed to allocate memory for result");
        return;
    }

    if (total_elements == 0)
    {
        data[0] = -FLT_MAX;
        from_data(out, data); // Set output data even for empty tensor
        // SAFE_FREE(&data, free); // data is owned by out
        return;
    }

    bool is_a_contiguous = is_contiguous(a);

    float max_val = -FLT_MAX; // Initialize with smallest possible float

    if (is_a_contiguous)
    {
        __m256 max_vec = _mm256_set1_ps(-FLT_MAX);
        size_t i = 0;

        for (; i + SIMD_WIDTH - 1 < total_elements; i += SIMD_WIDTH)
        {
            __m256 vec_a = _mm256_loadu_ps(a->data->data + i);
            max_vec = _mm256_max_ps(max_vec, vec_a);
        }

        // Horizontal max of SIMD vector
        __m128 vlow = _mm256_castps256_ps128(max_vec);
        __m128 vhigh = _mm256_extractf128_ps(max_vec, 1);
        vlow = _mm_max_ps(vlow, vhigh);
        vlow = _mm_max_ps(vlow, _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(2, 3, 0, 1)));
        vlow = _mm_max_ps(vlow, _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(1, 0, 3, 2)));
        max_val = _mm_cvtss_f32(vlow);

        for (; i < total_elements; ++i)
        {
            max_val = fmaxf(max_val, a->data->data[i]);
        }
    }
    else
    {
        int* coords = (int*) malloc(a->ndim * sizeof(int));
        if (!coords)
        {
            LOG_ERROR("max_full_op: Failed to allocate memory for coords.");
            SAFE_FREE(&data, free); // Free 'data' on failure
            return;
        }

        memset(coords, 0, a->ndim * sizeof(int));

        // First element is max_val
        size_t first_flat_idx = 0;
        for (int d = 0; d < a->ndim; ++d)
        {
            first_flat_idx += (size_t) coords[d] * a->strides[d];
        }
        max_val = a->data->data[first_flat_idx]; // Initialize max_val with the first element

        // Iterate from the second element
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

        for (size_t elem = 1; elem < total_elements; ++elem) // Start from second element
        {
            size_t flat_idx = 0;
            for (int d = 0; d < a->ndim; ++d)
            {
                flat_idx += (size_t) coords[d] * a->strides[d];
            }

            max_val = fmaxf(max_val, a->data->data[flat_idx]);

            carry = 1;
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

    data[0] = max_val;
    from_data(out, data);
    SAFE_FREE(&data, free);
}
