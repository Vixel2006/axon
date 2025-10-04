#include "utils.h"
#include <stdlib.h>
#include <string.h>

#include "logger.h"
#include "ops/movement_ops.h"

#define MAX_NDIM 32

#define COMPUTE_OFFSETS(linear_idx, tensor_for_shape, str_a, str_b, str_out, off_a, off_b,         \
                        off_out)                                                                   \
    do                                                                                             \
    {                                                                                              \
        int idx = linear_idx;                                                                      \
        off_a = 0;                                                                                 \
        off_b = 0;                                                                                 \
        off_out = 0;                                                                               \
        for (int d = tensor_for_shape->ndim - 1; d >= 0; --d)                                      \
        {                                                                                          \
            int coord = idx % tensor_for_shape->shape[d];                                          \
            idx /= tensor_for_shape->shape[d];                                                     \
            off_a += coord * str_a[d];                                                             \
            off_b += coord * str_b[d];                                                             \
            off_out += coord * str_out[d];                                                         \
        }                                                                                          \
    } while (0)

void view_op(Tensor* in, Tensor* out, int* shape, int ndim)
{
    LOG_INFO("OP: view_op: Creating view from Tensor %p (ndim=%d)", (void*) in, ndim);
    borrow(out, in->data, in->grad);
}

void unsqueeze_op(Tensor* in, Tensor* out, int dim)
{
    LOG_INFO("OP: unsqueeze_op: Unsqueezing Tensor %p at dimension %d", (void*) in, dim);

    if (dim < 0 || dim > in->ndim)
    {
        LOG_ERROR("unsqueeze_op: Invalid dimension %d for unsqueeze operation "
                  "(ndim=%d).",
                  dim, in->ndim);
        return;
    }

    for (int i = 0; i < dim; ++i)
    {
        out->strides[i] = in->strides[i];
    }
    for (int i = dim + 1; i < out->ndim; ++i)
    {
        out->strides[i] = in->strides[i - 1];
    }
    if (dim < out->ndim - 1)
    {
        out->strides[dim] = out->strides[dim + 1];
    }
    else
    {
        out->strides[dim] = out->strides[dim - 1];
    }

    borrow(out, in->data, in->grad);
}

void squeeze_op(Tensor* in, Tensor* out, int dim)
{
    LOG_INFO("OP: squeeze_op: Squeezing Tensor %p at dimension %d", (void*) in, dim);

    if (dim < 0 || dim >= in->ndim)
    {
        LOG_ERROR("squeeze_op: Invalid dimension %d for squeeze operation "
                  "(ndim=%d).",
                  dim, in->ndim);
        return;
    }

    if (in->shape[dim] != 1)
    {
        LOG_ERROR("Cannot squeeze dimension %d, size != 1", dim);
        return;
    }

    for (int i = 0; i < out->ndim; ++i)
        out->strides[i] = (i < dim) ? in->strides[i] : in->strides[i + 1];

    borrow(out, in->data, in->grad);
}

void transpose_op(Tensor* in, Tensor* out, int N, int M)
{
    if (!in || !out)
    {
        LOG_ERROR("transpose_op: Input or output tensor is NULL.");
        return;
    }

    LOG_INFO("OP: transpose_op: Transposing Tensor %p (dims %d, %d)", (void*) in, N, M);

    if (N < 0 || N >= in->ndim || M < 0 || M >= in->ndim || N == M)
    {
        LOG_ERROR("transpose_op: Invalid dimensions N=%d or M=%d for transpose operation "
                  "(ndim=%d). N and M must be within bounds and different.",
                  N, M, in->ndim);
        return;
    }

    for (int i = 0; i < out->ndim; ++i)
    {
        if (i == N)
        {
            out->strides[i] = in->strides[M];
        }
        else if (i == M)
        {
            out->strides[i] = in->strides[N];
        }
        else
        {
            out->strides[i] = in->strides[i];
        }
    }

    borrow(out, in->data, in->grad);
}

// WARNING: Assumes in->ndim == out->ndim
void expand_op(Tensor* in, Tensor* out, const int* shape)
{
    LOG_INFO("OP: expand_op: Expanding Tensor %p", (void*) in);

    for (int i = 0; i < in->ndim; ++i)
    {
        if (in->shape[i] != 1 && in->shape[i] != out->shape[i])
        {
            LOG_ERROR("expand_op: Cannot expand dimension %d from %d to %d. "
                      "Dimension must be 1 or match target size.",
                      i, in->shape[i], out->shape[i]);
            free(out->shape);
            free(out->strides);
            return;
        }
        out->strides[i] = (in->shape[i] == 1) ? 0 : in->strides[i];
    }
    borrow(out, in->data, in->grad);
}

void broadcast_op(Tensor* in, Tensor* out, int ndim, const int* shape)
{
    LOG_INFO("OP: broadcast_op: Broadcasting Tensor %p to ndim=%d", (void*) in, ndim);
    if (!in || !out || !shape)
    {
        LOG_ERROR("broadcast_op ERROR: Input tensor, output tensor, or shape "
                  "array is NULL! in=%p, out=%p, shape=%p",
                  (void*) in, (void*) out, (void*) shape);
        return;
    }

    int in_dim = in->ndim - 1;
    for (int i = ndim - 1; i >= 0; --i)
    {
        if (in_dim >= 0)
        {
            if (in->shape[in_dim] == shape[i])
            {
                out->strides[i] = in->strides[in_dim];
            }
            else if (in->shape[in_dim] == 1)
            {
                out->strides[i] = 0;
            }
            else
            {
                LOG_ERROR("broadcast_op: Cannot broadcast dimension %d from %d to "
                          "%d. Dimension must be 1 or match target size.",
                          in_dim, in->shape[in_dim], shape[i]);
                free(out->shape);
                free(out->strides);
                return;
            }
            in_dim--;
        }
        else
        {
            out->strides[i] = 0;
        }
    }

    borrow(out, in->data, in->grad);
}

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

    from_data_cpu(out, data);
    free(cum_sizes);
    free(data);
}
