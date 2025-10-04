#include "utils.h"
#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "logger.h"
#include "ops/reduction_ops.h"

#define SIMD_WIDTH 8

void sum_op_cpu(Tensor* a, Tensor* out, int axis, bool keepdim)
{
    LOG_INFO("OP: sum_op: Performing sum reduction along axis %d (keepdim=%d)", axis, keepdim);

    if (axis < 0 || axis >= a->ndim)
    {
        LOG_ERROR("sum_op: Invalid axis %d for tensor with %d dimensions", axis, a->ndim);
        return;
    }

    size_t out_size = numel(out->shape, out->ndim);
    float* data = (float*) malloc(sizeof(float) * out_size);
    if (!data)
    {
        LOG_ERROR("sum_op: Failed to allocate memory for output data");
        return;
    }
    memset(data, 0, out_size * sizeof(float));

    size_t in_size = numel(a->shape, a->ndim);

    int* a_coords = (int*) malloc(a->ndim * sizeof(int));
    if (!a_coords)
    {
        LOG_ERROR("sum_op: Failed to allocate memory for a_coords.");
        SAFE_FREE(&data, free);
        return;
    }

    int* out_coords = NULL;
    if (out->ndim > 0) // Allocate only if out->ndim is not 0 (scalar output)
    {
        out_coords = (int*) malloc(out->ndim * sizeof(int));
        if (!out_coords)
        {
            LOG_ERROR("sum_op: Failed to allocate memory for out_coords.");
            SAFE_FREE(&data, free);
            SAFE_FREE(&a_coords, free);
            return;
        }
    }

    for (size_t i = 0; i < in_size; ++i)
    {
        size_t in_offset = 0;
        size_t out_offset = 0;

        // Calculate coordinates in input tensor
        size_t temp_in_i = i;
        for (int d = a->ndim - 1; d >= 0; --d)
        {
            a_coords[d] = temp_in_i % a->shape[d];
            temp_in_i /= a->shape[d];
        }

        // Calculate coordinates in output tensor and its offset
        if (out->ndim == 0) // Scalar output
        {
            out_offset = 0;
        }
        else
        {
            int current_out_dim = 0;
            for (int d = 0; d < a->ndim; ++d)
            {
                if (d == axis)
                {
                    if (keepdim)
                    {
                        out_coords[current_out_dim++] = 0;
                    }
                }
                else
                {
                    out_coords[current_out_dim++] = a_coords[d];
                }
            }

            for (int d = 0; d < out->ndim; ++d)
            {
                out_offset += (size_t) out_coords[d] * out->strides[d];
            }
        }

        // Calculate flat offset for input
        for (int d = 0; d < a->ndim; ++d)
        {
            in_offset += (size_t) a_coords[d] * a->strides[d];
        }

        data[out_offset] += a->data->data[in_offset];
    }

    SAFE_FREE(&a_coords, free);
    SAFE_FREE(&out_coords, free);

    from_data_cpu(out, data); // Assuming from_data_cpu takes ownership of 'data'
    SAFE_FREE(&data, free);   // 'data' is now owned by 'out' if from_data_cpu works this way
}

void mean_op_cpu(Tensor* a, Tensor* out, int axis, bool keepdim)
{
    LOG_INFO("OP: mean_op: Performing mean reduction along axis %d (keepdim=%d)", axis, keepdim);

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
            // However, from_data_cpu typically makes 'out' contiguous or handles data pointers.
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

void max_op_cpu(Tensor* a, Tensor* out, int axis, bool keepdim)
{
    LOG_INFO("OP: max_op: Performing max reduction along axis %d (keepdim=%d)", axis, keepdim);

    if (axis < 0 || axis >= a->ndim)
    {
        LOG_ERROR("max_op: Invalid axis %d for tensor with %d dimensions", axis, a->ndim);
        return;
    }

    size_t out_size = numel(out->shape, out->ndim);
    float* data = (float*) malloc(sizeof(float) * out_size);
    if (!data)
    {
        LOG_ERROR("max_op: Failed to allocate memory for output data");
        return;
    }
    for (size_t i = 0; i < out_size; i++)
    {
        data[i] = -FLT_MAX;
    }

    // Calculate dimensions for iteration
    size_t outer_size = 1;
    for (int i = 0; i < axis; i++)
    {
        outer_size *= a->shape[i];
    }

    size_t reduction_size = a->shape[axis];

    size_t inner_size = 1;
    for (int i = axis + 1; i < a->ndim; i++)
    {
        inner_size *= a->shape[i];
    }

    size_t reduction_stride = a->strides[axis];

    int* a_coords = (int*) malloc(a->ndim * sizeof(int));
    if (!a_coords)
    {
        LOG_ERROR("max_op: Failed to allocate memory for a_coords.");
        SAFE_FREE(&data, free);
        return;
    }
    int* out_coords = NULL;
    if (out->ndim > 0)
    {
        out_coords = (int*) malloc(out->ndim * sizeof(int));
        if (!out_coords)
        {
            LOG_ERROR("max_op: Failed to allocate memory for out_coords.");
            SAFE_FREE(&data, free);
            SAFE_FREE(&a_coords, free);
            return;
        }
    }

    // Process each outer x inner slice
    for (size_t outer = 0; outer < outer_size; outer++)
    {
        for (size_t inner = 0; inner < inner_size; inner++)
        {
            // Reconstruct coordinates up to 'axis' and after 'axis'
            size_t input_base_offset = 0;
            size_t output_target_offset = 0;

            // Compute current outer_coords
            size_t temp_outer = outer;
            for (int d = axis - 1; d >= 0; d--)
            {
                a_coords[d] = temp_outer % a->shape[d];
                temp_outer /= a->shape[d];
                input_base_offset += (size_t) a_coords[d] * a->strides[d];

                if (out_coords &&
                    d < out->ndim) // Map to output coordinates if out_coords allocated
                {
                    // If keepdim, the axis dimension will be 0, otherwise it's skipped
                    int out_d_idx = d;
                    if (!keepdim && d >= axis)
                    {
                        out_d_idx--; // Adjust if reduction removed a dimension before it
                    }
                    if (out_d_idx >= 0 && out_d_idx < out->ndim)
                    {
                        out_coords[out_d_idx] = a_coords[d];
                    }
                }
            }
            // For the reduced axis itself
            a_coords[axis] = 0; // Temporary, will iterate over this axis
            if (out_coords && keepdim)
            {
                int out_d_idx = axis;
                if (out_d_idx < out->ndim)
                {
                    out_coords[out_d_idx] = 0; // Reduced dimension becomes 0 in output
                }
            }

            // Compute current inner_coords
            size_t temp_inner = inner;
            for (int d = a->ndim - 1; d > axis; d--)
            {
                a_coords[d] = temp_inner % a->shape[d];
                temp_inner /= a->shape[d];
                input_base_offset += (size_t) a_coords[d] * a->strides[d];

                if (out_coords && d < out->ndim)
                {
                    int out_d_idx = d;
                    if (!keepdim && d > axis)
                    {
                        out_d_idx--; // Adjust for removed dimension
                    }
                    if (keepdim && d == axis + 1 && axis + 1 < out->ndim)
                    {
                        // The output index for dimensions after 'axis' will be offset by 1 if
                        // keepdim is true Example: (2,3,4) axis=1, keepdim=true -> (2,1,4) Input
                        // d=2 (coord a_coords[2]) maps to output d=2 (out_coords[2]) But if not
                        // keepdim: (2,3,4) axis=1, keepdim=false -> (2,4) Input d=2 (coord
                        // a_coords[2]) maps to output d=1 (out_coords[1]) Simplified map:
                    }
                    if (out_d_idx >= 0 && out_d_idx < out->ndim)
                    {
                        out_coords[out_d_idx] = a_coords[d];
                    }
                }
            }

            // Correctly determine output_target_offset based on current out_coords
            if (out->ndim == 0) // Scalar output case
            {
                output_target_offset = 0;
            }
            else
            {
                int current_out_dim_idx = 0;
                for (int d = 0; d < a->ndim; ++d)
                {
                    if (d == axis)
                    {
                        if (keepdim)
                        {
                            output_target_offset += (size_t) 0 * out->strides[current_out_dim_idx];
                            current_out_dim_idx++;
                        }
                    }
                    else
                    {
                        output_target_offset +=
                            (size_t) a_coords[d] * out->strides[current_out_dim_idx];
                        current_out_dim_idx++;
                    }
                }
            }

            // Find maximum along the specified axis
            float max_val = -FLT_MAX;

            // This SIMD path only works for reduction_stride == 1
            if (reduction_stride == 1 && reduction_size >= SIMD_WIDTH && is_contiguous(a))
            {
                __m256 max_vec = _mm256_set1_ps(-FLT_MAX);
                size_t i = 0;

                for (; i + SIMD_WIDTH - 1 < reduction_size; i += SIMD_WIDTH)
                {
                    __m256 vec_a = _mm256_loadu_ps(a->data->data + input_base_offset + i);
                    max_vec = _mm256_max_ps(max_vec, vec_a);
                }

                // Horizontal max of SIMD vector
                __m128 vlow = _mm256_castps256_ps128(max_vec);
                __m128 vhigh = _mm256_extractf128_ps(max_vec, 1);
                vlow = _mm_max_ps(vlow, vhigh);
                vlow = _mm_max_ps(vlow, _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(2, 3, 0, 1)));
                vlow = _mm_max_ps(vlow, _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(1, 0, 3, 2)));
                max_val = _mm_cvtss_f32(vlow);

                // Handle remaining elements
                for (; i < reduction_size; i++)
                {
                    max_val = fmaxf(max_val, a->data->data[input_base_offset + i]);
                }
            }
            else
            {
                // Scalar path
                for (size_t i = 0; i < reduction_size; i++)
                {
                    max_val =
                        fmaxf(max_val, a->data->data[input_base_offset + i * reduction_stride]);
                }
            }

            data[output_target_offset] = max_val;
        }
    }
    SAFE_FREE(&a_coords, free);
    SAFE_FREE(&out_coords, free);

    from_data_cpu(out, data);
    SAFE_FREE(&data, free);
}

void sum_full_op_cpu(Tensor* a, Tensor* out)
{
    LOG_INFO("OP: sum_full_op: Performing full sum reduction");

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
        from_data_cpu(out, total_sum_ptr);
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

    from_data_cpu(out, total_sum_ptr);
    SAFE_FREE(&total_sum_ptr, free);
}

void mean_full_op_cpu(Tensor* a, Tensor* out)
{
    LOG_INFO("OP: mean_full_op: Performing full mean reduction");

    sum_full_op_cpu(a, out); // This will set out->data to the sum

    if (!out->data || !out->data->data)
    {
        LOG_ERROR("mean_full_op: sum_full_op failed to produce valid output data.");
        return;
    }

    size_t total_elements = numel(a->shape, a->ndim);
    // Directly modify out->data->data as it's already a single float array from sum_full_op
    // We assume out->data->data points to a malloc'd float[1] from sum_full_op
    // and from_data_cpu reallocates/manages its buffer on subsequent calls.

    if (total_elements > 0)
    {
        out->data->data[0] /= total_elements;
    }
    else
    {
        out->data->data[0] = 0.0f; // Or NaN, depending on desired behavior for empty mean
    }
    // No need to call from_data_cpu again if we modified the data in place,
    // and out->data->data already points to the correct buffer from sum_full_op.
    // If from_data_cpu always creates a new buffer, then a copy/re-assignment would be needed.
    // Assuming out->data->data is the buffer we operate on.
}

void max_full_op_cpu(Tensor* a, Tensor* out)
{
    LOG_INFO("OP: max_full_op: Performing full max reduction");

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
        from_data_cpu(out, data); // Set output data even for empty tensor
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
    from_data_cpu(out, data);
    SAFE_FREE(&data, free);
}
