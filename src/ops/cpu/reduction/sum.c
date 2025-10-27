#include "ops/cpu/reduction.h"

void sum_op_cpu(Tensor* a, Tensor* out, int axis, bool keepdim)
{
    LOG_INFO("sum_op_cpu: Entering function with axis=%d, keepdim=%d", axis, keepdim);

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

    from_data(out, data);   // Assuming from_data takes ownership of 'data'
    SAFE_FREE(&data, free); // 'data' is now owned by 'out' if from_data works this way
}
