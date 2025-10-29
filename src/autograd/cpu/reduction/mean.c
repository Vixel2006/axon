#include "autograd/cpu/reduction/common.h"

void mean_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("mean_grad_op_cpu: Entering function with n_prev=%d", n_prev);

    if (!out || !out->grad || !out->grad->data->data || !prev)
    {
        LOG_ERROR("mean_grad_op: Output tensor, output gradient, or previous tensors array is "
                  "NULL! out=%p, out->grad=%p, prev=%p",
                  (void*) out, out ? (void*) out->grad : NULL, (void*) prev);
        return;
    }

    if (n_prev != 1)
    {
        LOG_ERROR("mean_grad_op: Invalid number of previous tensors: %d. Expected 1.", n_prev);
        return;
    }

    if (!prev[0])
    {
        LOG_ERROR("mean_grad_op: Previous tensor is NULL! prev[0]=%p", (void*) prev[0]);
        return;
    }

    Tensor* in = prev[0];

    if (!in->requires_grad)
    {
        return;
    }

    if (!in->grad || !in->grad->data->data)
    {
        LOG_ERROR("mean_grad_op: Input tensor requires grad but its grad data is NULL!");
        return;
    }

    int reduced_dim = get_reduced_dim(in->shape, out->shape, in->ndim, out->ndim);

    if (reduced_dim == -1)
    {
        // No reduction case
        int size = numel(in->shape, in->ndim);
        if (!is_contiguous(in) || !is_contiguous(out))
        {
            int* in_strides = in->strides;
            int* out_strides = out->strides;
            int* shape = in->shape;
            int ndim = in->ndim;

            for (int linear = 0; linear < size; ++linear)
            {
                int idx = linear;
                int in_offset = 0, out_offset = 0;

                for (int d = ndim - 1; d >= 0; --d)
                {
                    int coord = idx % shape[d];
                    idx /= shape[d];

                    in_offset += coord * in_strides[d];
                    out_offset += coord * out_strides[d];
                }
                in->grad->data->data[in_offset] += out->grad->data->data[out_offset];
            }
        }
        else
        {
            int i = 0;
            for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
            {
                __m256 in_grad = _mm256_loadu_ps(in->grad->data->data + i);
                __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
                __m256 new_in_grad = _mm256_add_ps(in_grad, dout);
                _mm256_storeu_ps(in->grad->data->data + i, new_in_grad);
            }
            for (; i < size; ++i)
            {
                in->grad->data->data[i] += out->grad->data->data[i];
            }
        }
        return;
    }

    // Reduction case
    int in_size = numel(in->shape, in->ndim);
    int* in_strides = in->strides;
    int* in_shape = in->shape;
    int in_ndim = in->ndim;

    int reduce_size = in->shape[reduced_dim];

    if (!is_contiguous(in) || !is_contiguous(out))
    {
        int* in_coords = malloc(in_ndim * sizeof(int));
        if (!in_coords)
        {
            LOG_ERROR("mean_grad_op: Failed to allocate coordinates");
            return;
        }

        for (int in_linear_idx = 0; in_linear_idx < in_size; ++in_linear_idx)
        {
            // Convert linear index to coordinates
            int temp_idx = in_linear_idx;
            for (int d = in_ndim - 1; d >= 0; --d)
            {
                in_coords[d] = temp_idx % in_shape[d];
                temp_idx /= in_shape[d];
            }

            // Calculate input offset
            int in_offset = 0;
            for (int d = 0; d < in_ndim; ++d)
            {
                in_offset += in_coords[d] * in_strides[d];
            }

            // Calculate output offset using helper
            int out_offset = 0;
            map_in_coords_to_out_offset(in_coords, in_ndim, reduced_dim, out, &out_offset);

            in->grad->data->data[in_offset] += out->grad->data->data[out_offset] / reduce_size;
        }

        free(in_coords);
    }
    else
    {
        // Contiguous path
        int num_batches = numel(out->shape, out->ndim);

        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx)
        {
            float grad = out->grad->data->data[batch_idx];

            __m256 grad_vec = _mm256_set1_ps(grad / reduce_size);

            int base_offset = batch_idx * reduce_size;

            int i = 0;
            for (; i + SIMD_WIDTH - 1 < reduce_size; i += SIMD_WIDTH)
            {
                __m256 data_vec = _mm256_loadu_ps(in->grad->data->data + base_offset + i);
                data_vec = _mm256_add_ps(data_vec, grad_vec);
                _mm256_storeu_ps(in->grad->data->data + base_offset + i, data_vec);
            }

            for (; i < reduce_size; ++i)
            {
                in->grad->data->data[base_offset + i] += grad / reduce_size;
            }
        }
    }
    LOG_INFO("mean_grad_op_cpu: Exiting function.");
}
