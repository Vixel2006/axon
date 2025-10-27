#include "ops/cpu/reduction.h"

void max_op_cpu(Tensor* a, Tensor* out, int axis, bool keepdim)
{
    LOG_INFO("max_op_cpu: Entering function with axis=%d, keepdim=%d", axis, keepdim);

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

    from_data(out, data);
    SAFE_FREE(&data, free);
}
