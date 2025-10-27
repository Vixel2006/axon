#include "ops/cpu/init.h" // For from_data
#include "ops/cpu/movement.h"

void concat_op_cpu(Tensor** in, Tensor* out, int num_tensors, int axis)
{
    if (!in || !out || num_tensors < 1)
    {
        LOG_ERROR("concat_op ERROR: Input tensors array is NULL, output tensor is NULL, "
                  "or num_tensors < 1! in=%p, out=%p, num_tensors=%d",
                  (void*) in, (void*) out, num_tensors);
        return;
    }

    LOG_INFO("OP: concat_op: Concatenating %d tensors along axis %d", num_tensors, axis);

    int ndim = in[0]->ndim;
    if (axis < 0 || axis >= ndim)
    {
        LOG_ERROR("concat_op: Invalid axis %d (ndim=%d).", axis, ndim);
        return;
    }

    for (int i = 1; i < num_tensors; ++i)
    {
        if (in[i]->ndim != ndim)
        {
            LOG_ERROR("concat_op: Input tensors must have the same ndim (%d != %d).", in[i]->ndim,
                      ndim);
            return;
        }
        for (int d = 0; d < ndim; ++d)
        {
            if (d != axis && in[i]->shape[d] != in[0]->shape[d])
            {
                LOG_ERROR("concat_op: Input shapes mismatch on dim %d (%d != %d).", d,
                          in[i]->shape[d], in[0]->shape[d]);
                return;
            }
        }
    }

    if (out->ndim != ndim)
    {
        LOG_ERROR("concat_op: Output ndim mismatch (%d != %d).", out->ndim, ndim);
        return;
    }

    int sum_axis_size = 0;
    int* cum_sizes = malloc((num_tensors + 1) * sizeof(int));
    if (!cum_sizes)
    {
        LOG_ERROR("concat_op: Memory allocation failed for cum_sizes.");
        return;
    }
    cum_sizes[0] = 0;
    for (int i = 0; i < num_tensors; ++i)
    {
        sum_axis_size += in[i]->shape[axis];
        cum_sizes[i + 1] = sum_axis_size;
    }

    if (out->shape[axis] != sum_axis_size)
    {
        LOG_ERROR("concat_op: Output shape on axis %d mismatch (expected %d, got %d).", axis,
                  sum_axis_size, out->shape[axis]);
        free(cum_sizes);
        return;
    }
    for (int d = 0; d < ndim; ++d)
    {
        if (d != axis && out->shape[d] != in[0]->shape[d])
        {
            LOG_ERROR("concat_op: Output shape mismatch on dim %d (%d != %d).", d, out->shape[d],
                      in[0]->shape[d]);
            free(cum_sizes);
            return;
        }
    }

    int total_size = numel(out->shape, out->ndim);
    float* data = malloc(sizeof(float) * total_size);
    if (!data)
    {
        LOG_ERROR("concat_op: Memory allocation failed for output data.");
        free(cum_sizes);
        return;
    }

    for (int t = 0; t < num_tensors; ++t)
    {
        Tensor* src = in[t];
        int in_size = numel(src->shape, src->ndim);
        int cum = cum_sizes[t];
        int off_in, off_b, off_out;
        int* dummy_str = src->strides;
        for (int linear_idx = 0; linear_idx < in_size; ++linear_idx)
        {
            COMPUTE_OFFSETS(linear_idx, src, src->strides, dummy_str, out->strides, off_in, off_b,
                            off_out);
            off_out += cum * out->strides[axis];
            data[off_out] = src->data->data[off_in];
        }
    }

    from_data(out, data);
    free(cum_sizes);
    free(data);
}
