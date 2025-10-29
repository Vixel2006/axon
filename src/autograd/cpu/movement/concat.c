#include "autograd/autograd_movement.h"
#include "logger.h"
#include "tensor.h"
#include "utils.h"
#include <stdlib.h>

#define SIMD_WIDTH 8
#define MAX_DIMS 32

void concat_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    ConcatExtras* concat_extras = (ConcatExtras*) extras;
    int axis = concat_extras->axis;

    LOG_INFO("concat_grad_op_cpu: Entering function with out.numel=%d, n_prev=%d, axis=%d",
             numel(out->shape, out->ndim), n_prev, axis);

    if (out->grad == NULL || out->grad->data->data == NULL || out->grad->data->data == NULL)
    {
        LOG_WARN("concat_grad_op: Output tensor has no gradient data, skipping backward pass for "
                 "concat.");
        return;
    }

    int outer_size = 1;
    for (int i = 0; i < axis; ++i)
        outer_size *= out->shape[i];

    int inner_size = 1;
    for (int i = axis + 1; i < out->ndim; ++i)
        inner_size *= out->shape[i];

    int out_concat_axis_size = out->shape[axis];
    int offset_in_axis = 0;

    for (int i = 0; i < n_prev; ++i)
    {
        Tensor* current_prev = prev[i];

        if (current_prev->requires_grad)
        {
            if (current_prev->grad == NULL || current_prev->grad->data->data == NULL ||
                current_prev->grad->data->data == NULL)
            {
                LOG_WARN("concat_grad_op: prev tensor %d has no gradient data buffer, skipping.",
                         i);
                offset_in_axis += current_prev->shape[axis];
                continue;
            }

            int N = numel(current_prev->shape, current_prev->ndim);
            int prev_concat_axis_size = current_prev->shape[axis];

            if (is_contiguous(current_prev))
            {
                // Here comes the contiguous path with SIMD
                for (int j = 0; j < N; ++j)
                {
                    int outer_i = j / (prev_concat_axis_size * inner_size);
                    int remainder = j % (prev_concat_axis_size * inner_size);
                    int out_idx = outer_i * (out_concat_axis_size * inner_size) +
                                  (offset_in_axis * inner_size) + remainder;
                    current_prev->grad->data->data[j] += out->grad->data->data[out_idx];
                }
            }
            else
            {
                // Here comes the uncontiguous path without SIMD
                if (current_prev->ndim > MAX_DIMS)
                {
                    LOG_ERROR("concat_grad_op: Tensor dimensions %d exceed MAX_DIMS %d",
                              current_prev->ndim, MAX_DIMS);
                    exit(30);
                }
                for (int j = 0; j < N; ++j)
                {
                    int coords[MAX_DIMS];
                    int temp_j = j;

                    for (int d = current_prev->ndim - 1; d >= 0; --d)
                    {
                        coords[d] = temp_j % current_prev->shape[d];
                        temp_j /= current_prev->shape[d];
                    }

                    int prev_idx = 0;
                    for (int d = 0; d < current_prev->ndim; ++d)
                    {
                        prev_idx += coords[d] * current_prev->strides[d];
                    }

                    int outer_i = j / (prev_concat_axis_size * inner_size);
                    int remainder = j % (prev_concat_axis_size * inner_size);
                    int out_idx = outer_i * (out_concat_axis_size * inner_size) +
                                  (offset_in_axis * inner_size) + remainder;

                    current_prev->grad->data->data[prev_idx] += out->grad->data->data[out_idx];
                }
            }
        }

        offset_in_axis += current_prev->shape[axis];
    }

    LOG_INFO("concat_grad_op: Exiting function.");
}
