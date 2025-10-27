#include "autograd/autograd_reduction.h"
#include "logger.h"
#include "utils.h"
#include <immintrin.h>
#include <stdlib.h>

#define SIMD_WIDTH 8

// Forward declaration
void max_full_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);

// Helper function to properly map input indices to output indices
static void map_in_coords_to_out_offset(int* in_coords, int in_ndim, int reduced_dim, Tensor* out,
                                        int* out_offset_result)
{
    int out_idx = 0;
    for (int d = 0; d < in_ndim; ++d)
    {
        if (d != reduced_dim)
        {
            // This dimension exists in output
            int out_coord = in_coords[d];
            // Find which output dimension this maps to
            int out_d = d;
            if (d > reduced_dim && out->ndim < in_ndim)
            {
                out_d = d - 1; // Adjust for removed dimension
            }
            if (out_d < out->ndim)
            {
                out_idx += out_coord * out->strides[out_d];
            }
        }
    }
    *out_offset_result = out_idx;
}

void sum_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("sum_grad_op_cpu: Entering function with n_prev=%d", n_prev);

    if (!out || !out->grad || !out->grad->data->data || !prev)
    {
        LOG_ERROR("sum_grad_op: Output tensor, output gradient, or previous tensors array is NULL! "
                  "out=%p, out->grad=%p, prev=%p",
                  (void*) out, out ? (void*) out->grad : NULL, (void*) prev);
        exit(31);
    }

    if (n_prev != 1)
    {
        LOG_ERROR("sum_grad_op: Invalid number of previous tensors: %d. Expected 1.", n_prev);
        exit(32);
    }

    if (!prev[0])
    {
        LOG_ERROR("sum_grad_op: Previous tensor is NULL! prev[0]=%p", (void*) prev[0]);
        exit(33);
    }

    Tensor* in = prev[0];

    if (!in->requires_grad)
    {
        return;
    }

    if (!in->grad || !in->grad->data->data)
    {
        LOG_ERROR("sum_grad_op: Input tensor requires grad but its grad data is NULL!");
        exit(34);
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

    if (!is_contiguous(in) || !is_contiguous(out))
    {
        int* in_coords = malloc(in_ndim * sizeof(int));
        if (!in_coords)
        {
            LOG_ERROR("sum_grad_op: Failed to allocate coordinates");
            exit(35);
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

            in->grad->data->data[in_offset] += out->grad->data->data[out_offset];
        }

        free(in_coords);
    }
    else
    {
        // Contiguous path
        int reduce_size = in->shape[reduced_dim];
        int num_batches = numel(out->shape, out->ndim);

        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx)
        {
            float grad = out->grad->data->data[batch_idx];

            __m256 grad_vec = _mm256_set1_ps(grad);

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
                in->grad->data->data[base_offset + i] += grad;
            }
        }
    }
    LOG_INFO("sum_grad_op_cpu: Exiting function.");
}

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

void max_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("max_grad_op_cpu: Entering function with n_prev=%d", n_prev);

    if (!out || !out->grad || !out->grad->data->data || !prev)
    {
        LOG_ERROR("max_grad_op: Output tensor, output gradient, or previous tensors array is NULL! "
                  "out=%p, out->grad=%p, prev=%p",
                  (void*) out, out ? (void*) out->grad : NULL, (void*) prev);
        return;
    }

    if (n_prev != 1)
    {
        LOG_ERROR("max_grad_op: Invalid number of previous tensors: %d. Expected 1.", n_prev);
        return;
    }

    if (!prev[0])
    {
        LOG_ERROR("max_grad_op ERROR: Previous tensor is NULL! prev[0]=%p", (void*) prev[0]);
        return;
    }

    Tensor* in = prev[0];

    if (!in->requires_grad)
    {
        return;
    }

    if (!in->grad || !in->grad->data->data)
    {
        LOG_ERROR("max_grad_op ERROR: Input tensor requires grad but its grad data is NULL!");
        return;
    }

    LOG_INFO("DEBUG: max_grad_op: out->ndim = %d", out->ndim);

    // If output is a scalar, it's a full reduction. Delegate to max_full_grad_op.
    if (out->ndim == 0)
    {
        max_full_grad_op_cpu(out, prev, n_prev, extras);
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

    if (!is_contiguous(in) || !is_contiguous(out))
    {
        int* in_coords = malloc(in_ndim * sizeof(int));
        if (!in_coords)
        {
            LOG_ERROR("max_grad_op: Failed to allocate coordinates");
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

            if (in->data->data[in_offset] == out->data->data[out_offset])
            {
                in->grad->data->data[in_offset] += out->grad->data->data[out_offset];
            }
        }

        free(in_coords);
    }
    else
    {
        // Contiguous path
        int reduce_size = in->shape[reduced_dim];
        int num_batches = numel(out->shape, out->ndim);

        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx)
        {
            float grad = out->grad->data->data[batch_idx];
            float max = out->data->data[batch_idx];

            __m256 grad_vec = _mm256_set1_ps(grad);
            __m256 max_vec = _mm256_set1_ps(max);

            int base_offset = batch_idx * reduce_size;

            int i = 0;
            for (; i + SIMD_WIDTH - 1 < reduce_size; i += SIMD_WIDTH)
            {
                __m256 data_vec = _mm256_loadu_ps(in->data->data + base_offset + i);
                __m256 mask = _mm256_cmp_ps(data_vec, max_vec, _CMP_EQ_OQ);
                __m256 grad_contrib = _mm256_and_ps(grad_vec, mask);
                __m256 in_grad = _mm256_loadu_ps(in->grad->data->data + base_offset + i);
                __m256 new_grad = _mm256_add_ps(in_grad, grad_contrib);
                _mm256_storeu_ps(in->grad->data->data + base_offset + i, new_grad);
            }

            for (; i < reduce_size; ++i)
            {
                if (in->data->data[base_offset + i] == max)
                {
                    in->grad->data->data[base_offset + i] += grad;
                }
            }
        }
    }
    LOG_INFO("max_grad_op_cpu: Exiting function.");
}

void sum_full_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("sum_full_grad_op_cpu: Entering function with n_prev=%d", n_prev);

    if (!out || !out->grad || !out->grad->data->data || !prev)
    {
        LOG_ERROR("sum_full_grad_op ERROR: Output tensor, output gradient, or previous "
                  "tensors array is NULL! out=%p, out->grad=%p, prev=%p",
                  (void*) out, out ? (void*) out->grad : NULL, (void*) prev);
        return;
    }

    if (n_prev != 1)
    {
        LOG_ERROR("sum_full_grad_op ERROR: Invalid number of previous tensors: %d. "
                  "Expected 1.",
                  n_prev);
        return;
    }

    if (!prev[0])
    {
        LOG_ERROR("sum_full_grad_op ERROR: Previous tensor is NULL! prev[0]=%p", (void*) prev[0]);
        return;
    }

    Tensor* in = prev[0];

    if (!in->requires_grad)
    {
        return;
    }

    if (!in->grad || !in->grad->data->data)
    {
        LOG_ERROR("sum_full_grad_op ERROR: Input tensor requires grad but its grad data is NULL!");
        return;
    }

    float output_grad = out->grad->data->data[0];
    int in_size = numel(in->shape, in->ndim);

    if (is_contiguous(in))
    {
        __m256 grad_vec = _mm256_set1_ps(output_grad);

        int i = 0;
        for (; i + SIMD_WIDTH - 1 < in_size; i += SIMD_WIDTH)
        {
            __m256 in_grad = _mm256_loadu_ps(in->grad->data->data + i);
            __m256 new_grad = _mm256_add_ps(in_grad, grad_vec);
            _mm256_storeu_ps(in->grad->data->data + i, new_grad);
        }

        for (; i < in_size; ++i)
        {
            in->grad->data->data[i] += output_grad;
        }
    }
    else
    {
        int* in_strides = in->strides;
        int* in_shape = in->shape;
        int in_ndim = in->ndim;

        for (int linear_idx = 0; linear_idx < in_size; ++linear_idx)
        {
            int temp_idx = linear_idx;
            int in_offset = 0;

            for (int d = in_ndim - 1; d >= 0; --d)
            {
                int coord = temp_idx % in_shape[d];
                temp_idx /= in_shape[d];
                in_offset += coord * in_strides[d];
            }

            in->grad->data->data[in_offset] += output_grad;
        }
    }
    LOG_INFO("sum_full_grad_op_cpu: Exting function.");
}

void mean_full_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("mean_full_grad_op_cpu: Entering function with n_prev=%d", n_prev);

    if (!out || !out->grad || !out->grad->data->data || !prev)
    {
        LOG_ERROR("mean_full_grad_op ERROR: Output tensor, output gradient, or previous "
                  "tensors array is NULL! out=%p, out->grad=%p, prev=%p",
                  (void*) out, out ? (void*) out->grad : NULL, (void*) prev);
        return;
    }

    if (n_prev != 1)
    {
        LOG_ERROR("mean_full_grad_op ERROR: Invalid number of previous tensors: %d. "
                  "Expected 1.",
                  n_prev);
        return;
    }

    if (!prev[0])
    {
        LOG_ERROR("mean_full_grad_op ERROR: Previous tensor is NULL! prev[0]=%p", (void*) prev[0]);
        return;
    }

    Tensor* in = prev[0];

    if (!in->requires_grad)
    {
        return;
    }

    if (!in->grad || !in->grad->data->data)
    {
        LOG_ERROR("mean_full_grad_op ERROR: Input tensor requires grad but its grad data is NULL!");
        return;
    }

    float output_grad = out->grad->data->data[0];
    int in_size = numel(in->shape, in->ndim);

    float scaled_grad = output_grad / in_size;

    if (is_contiguous(in))
    {
        __m256 grad_vec = _mm256_set1_ps(scaled_grad);

        int i = 0;
        for (; i + SIMD_WIDTH - 1 < in_size; i += SIMD_WIDTH)
        {
            __m256 in_grad = _mm256_loadu_ps(in->grad->data->data + i);
            __m256 new_grad = _mm256_add_ps(in_grad, grad_vec);
            _mm256_storeu_ps(in->grad->data->data + i, new_grad);
        }

        for (; i < in_size; ++i)
        {
            in->grad->data->data[i] += scaled_grad;
        }
    }
    else
    {
        int* in_strides = in->strides;
        int* in_shape = in->shape;
        int in_ndim = in->ndim;

        for (int linear_idx = 0; linear_idx < in_size; ++linear_idx)
        {
            int temp_idx = linear_idx;
            int in_offset = 0;

            for (int d = in_ndim - 1; d >= 0; --d)
            {
                int coord = temp_idx % in_shape[d];
                temp_idx /= in_shape[d];
                in_offset += coord * in_strides[d];
            }

            in->grad->data->data[in_offset] += scaled_grad;
        }
    }
    LOG_INFO("mean_full_grad_op_cpu: Exiting function.");
}

void max_full_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("max_full_grad_op_cpu: Entering function with n_prev=%d", n_prev);

    if (!out || !out->grad || !out->grad->data->data || !prev)
    {
        LOG_ERROR("max_full_grad_op ERROR: Output tensor, output gradient, or previous "
                  "tensors array is NULL! out=%p, out->grad=%p, prev=%p",
                  (void*) out, out ? (void*) out->grad : NULL, (void*) prev);
        return;
    }

    if (n_prev != 1)
    {
        LOG_ERROR("max_full_grad_op ERROR: Invalid number of previous tensors: %d. "
                  "Expected 1.",
                  n_prev);
        return;
    }

    if (!prev[0])
    {
        LOG_ERROR("max_full_grad_op ERROR: Previous tensor is NULL! prev[0]=%p", (void*) prev[0]);
        return;
    }

    Tensor* in = prev[0];

    if (!in->requires_grad)
    {
        return;
    }

    if (!in->grad || !in->grad->data->data)
    {
        LOG_ERROR("max_full_grad_op ERROR: Input tensor requires grad but its grad data is NULL!");
        return;
    }

    float output_grad = out->grad->data->data[0];
    float max_val = out->data->data[0];
    int in_size = numel(in->shape, in->ndim);

    if (is_contiguous(in))
    {
        __m256 grad_vec = _mm256_set1_ps(output_grad);
        __m256 max_vec = _mm256_set1_ps(max_val);

        int i = 0;
        for (; i + SIMD_WIDTH - 1 < in_size; i += SIMD_WIDTH)
        {
            __m256 data_vec = _mm256_loadu_ps(in->data->data + i);
            __m256 mask = _mm256_cmp_ps(data_vec, max_vec, _CMP_EQ_OQ);
            __m256 grad_contrib = _mm256_and_ps(grad_vec, mask);
            __m256 in_grad = _mm256_loadu_ps(in->grad->data->data + i);
            __m256 new_grad = _mm256_add_ps(in_grad, grad_contrib);
            _mm256_storeu_ps(in->grad->data->data + i, new_grad);
        }

        for (; i < in_size; ++i)
        {
            if (in->data->data[i] == max_val)
            {
                in->grad->data->data[i] += output_grad;
            }
        }
    }
    else
    {
        int* in_strides = in->strides;
        int* in_shape = in->shape;
        int in_ndim = in->ndim;

        for (int linear_idx = 0; linear_idx < in_size; ++linear_idx)
        {
            int temp_idx = linear_idx;
            int in_offset = 0;

            for (int d = in_ndim - 1; d >= 0; --d)
            {
                int coord = temp_idx % in_shape[d];
                temp_idx /= in_shape[d];
                in_offset += coord * in_strides[d];
            }

            if (in->data->data[in_offset] == max_val)
            {
                in->grad->data->data[in_offset] += output_grad;
            }
        }
    }
    LOG_INFO("max_full_grad_op_cpu: Exiting function.");
}
