#include "ops/cpu/reduction.h"

void mean_full_op_cpu(Tensor* a, Tensor* out)
{
    LOG_INFO("mean_full_op_cpu: Entering function");

    sum_full_op_cpu(a, out); // This will set out->data to the sum

    if (!out->data || !out->data->data)
    {
        LOG_ERROR("mean_full_op: sum_full_op failed to produce valid output data.");
        return;
    }

    size_t total_elements = numel(a->shape, a->ndim);
    // Directly modify out->data->data as it's already a single float array from sum_full_op
    // We assume out->data->data points to a malloc'd float[1] from sum_full_op
    // and from_data reallocates/manages its buffer on subsequent calls.

    if (total_elements > 0)
    {
        out->data->data[0] /= total_elements;
    }
    else
    {
        out->data->data[0] = 0.0f; // Or NaN, depending on desired behavior for empty mean
    }
    // No need to call from_data again if we modified the data in place,
    // and out->data->data already points to the correct buffer from sum_full_op.
    // If from_data always creates a new buffer, then a copy/re-assignment would be needed.
    // Assuming out->data->data is the buffer we operate on.
}
