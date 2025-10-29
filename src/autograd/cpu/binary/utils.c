#include "autograd/cpu/binary/common.h"

void binary_grad_noncontig(Tensor* out, Tensor* a, Tensor* b, binary_grad_fn da_fn,
                           binary_grad_fn db_fn)
{
    int size = numel(out->shape, out->ndim);
    int out_ndim = out->ndim;
    int* out_shape = out->shape;
    int* out_strides = out->strides;

    int a_ndim = a->ndim;
    int* a_shape = a->shape;
    int* a_strides = a->strides;

    int b_ndim = b->ndim;
    int* b_shape = b->shape;
    int* b_strides = b->strides;

    float* a_data = a->data->data;
    float* b_data = b->data->data;
    float* out_grad = out->grad->data->data;
    float* a_grad = (a->requires_grad ? a->grad->data->data : NULL);
    float* b_grad = (b->requires_grad ? b->grad->data->data : NULL);

    for (int linear = 0; linear < size; ++linear)
    {
        int a_offset = 0, b_offset = 0, out_offset = 0;

        // Calculate offsets for 'out'
        int current_idx = linear;
        for (int d = out_ndim - 1; d >= 0; --d)
        {
            int coord = current_idx % out_shape[d];
            current_idx /= out_shape[d];
            out_offset += coord * out_strides[d];
        }

        // Calculate offsets for 'a'
        int a_current_linear_idx = linear;
        for (int d = out_ndim - 1; d >= 0; --d)
        {
            int out_coord = a_current_linear_idx % out_shape[d];
            a_current_linear_idx /= out_shape[d];

            int a_dim_idx = d - (out_ndim - a_ndim);
            if (a_dim_idx >= 0)
            {
                int a_coord = (a_shape[a_dim_idx] == 1) ? 0 : out_coord;
                a_offset += a_coord * a_strides[a_dim_idx];
            }
        }

        // Calculate offsets for 'b'
        int b_current_linear_idx = linear;
        for (int d = out_ndim - 1; d >= 0; --d)
        {
            int out_coord = b_current_linear_idx % out_shape[d];
            b_current_linear_idx /= out_shape[d];

            int b_dim_idx = d - (out_ndim - b_ndim);
            if (b_dim_idx >= 0)
            {
                int b_coord = (b_shape[b_dim_idx] == 1) ? 0 : out_coord;
                b_offset += b_coord * b_strides[b_dim_idx];
            }
        }

        float dout = out_grad[out_offset];
        float aval = a_data[a_offset];
        float bval = b_data[b_offset];
        if (a_grad) a_grad[a_offset] += da_fn(dout, aval, bval);
        if (b_grad) b_grad[b_offset] += db_fn(dout, aval, bval);
    }
}

void unary_grad_noncontig(Tensor* out, Tensor* a, float scalar, unary_grad_fn da_fn)
{
    if (!a->requires_grad) return;
    int size = numel(out->shape, out->ndim);
    int ndim = out->ndim;
    int* shape = out->shape;
    int* a_strides = a->strides;
    int* out_strides = out->strides;
    float* a_grad = a->grad->data->data;
    float* a_data = a->data->data;
    float* out_grad = out->grad->data->data;
    for (int linear = 0; linear < size; ++linear)
    {
        int idx = linear;
        int a_offset = 0, out_offset = 0;
        for (int d = ndim - 1; d >= 0; --d)
        {
            int coord = idx % shape[d];
            idx /= shape[d];
            a_offset += coord * a_strides[d];
            out_offset += coord * out_strides[d];
        }
        float dout = out_grad[out_offset];
        float aval = a_data[a_offset];
        a_grad[a_offset] += da_fn(dout, aval, scalar);
    }
}

