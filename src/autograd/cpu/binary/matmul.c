#include "autograd/cpu/binary/common.h"

void matmul_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("matmul_grad_op_cpu: Entering function with n_prev=%d", n_prev);

    if (!out || !prev)
    {
        LOG_ERROR(
            "matmul_grad_op: Output tensor or previous tensors array is NULL! out=%p, prev=%p",
            (void*) out, (void*) prev);
        assert(0 && "matmul_grad_op: Output tensor or previous tensors array is NULL!");
    }

    if (!out->grad->data->data)
    {
        LOG_ERROR("matmul_grad_op: Output gradient is NULL! out->grad=%p",
                  (void*) out->grad->data->data);
        assert(0 && "matmul_grad_op: Output gradient is NULL!");
    }

    if (n_prev != 2)
    {
        LOG_ERROR("matmul_grad_op: Invalid number of previous tensors: %d. Expected 2.", n_prev);
        assert(0 && "matmul_grad_op: Invalid number of previous tensors. Expected 2.");
    }

    if (!prev[0] || !prev[1])
    {
        LOG_ERROR("matmul_grad_op: One or both previous tensors are NULL! prev[0]=%p, prev[1]=%p",
                  (void*) prev[0], (void*) prev[1]);
        assert(0 && "matmul_grad_op: One or both previous tensors are NULL!");
    }

    Tensor* a = prev[0];
    Tensor* b = prev[1];

    if (a->ndim < 2 || b->ndim < 2 || out->ndim < 2)
    {
        LOG_ERROR("matmul_grad_op: All tensors must have at least 2 dimensions! a->ndim=%d, "
                  "b->ndim=%d, out->ndim=%d",
                  a->ndim, b->ndim, out->ndim);
        assert(0 && "matmul_grad_op: All tensors must have at least 2 dimensions!");
    }

    if (!a->shape || !b->shape || !out->shape)
    {
        LOG_ERROR("matmul_grad_op: One or more shape arrays are NULL!");
        assert(0 && "matmul_grad_op: One or more shape arrays are NULL!");
    }

    if (!a->strides || !b->strides || !out->strides)
    {
        LOG_ERROR("matmul_grad_op: One or more stride arrays are NULL!");
        assert(0 && "matmul_grad_op: One or more stride arrays are NULL!");
    }

    if (!a->data->data || !b->data->data)
    {
        LOG_ERROR("matmul_grad_op: One or more data arrays are NULL!");
        assert(0 && "matmul_grad_op: One or more data arrays are NULL!");
    }

    int N = a->shape[a->ndim - 2];
    int K = a->shape[a->ndim - 1];
    int M = b->shape[b->ndim - 1];

    if (a->shape[a->ndim - 1] != b->shape[b->ndim - 2])
    {
        LOG_ERROR("matmul_grad_op: Dimension mismatch for matrix multiplication! "
                  "a->shape[last]=%d, b->shape[second_last]=%d",
                  a->shape[a->ndim - 1], b->shape[b->ndim - 2]);
        assert(0 && "matmul_grad_op: Dimension mismatch for matrix multiplication!");
    }

    // Validate output dimensions match expected result
    if (out->shape[out->ndim - 2] != N || out->shape[out->ndim - 1] != M)
    {
        LOG_ERROR("matmul_grad_op: Output dimensions don't match expected result! Expected (%d, "
                  "%d), got (%d, %d)",
                  N, M, out->shape[out->ndim - 2], out->shape[out->ndim - 1]);
        assert(0 && "matmul_grad_op: Output dimensions don't match expected result!");
    }

    // Calculate total batch size (product of all batch dimensions)
    int batch_size = 1;
    for (int i = 0; i < out->ndim - 2; ++i)
    {
        int a_dim = (i < a->ndim - 2) ? a->shape[i] : 1;
        int b_dim = (i < b->ndim - 2) ? b->shape[i] : 1;
        int out_dim = (i < out->ndim - 2) ? out->shape[i] : 1;

        // Verify broadcasting compatibility
        if ((a_dim != 1 && b_dim != 1 && a_dim != b_dim) ||
            (a_dim != 1 && out_dim != 1 && a_dim != out_dim) ||
            (b_dim != 1 && out_dim != 1 && b_dim != out_dim))
        {
            LOG_ERROR(
                "matmul_grad_op: Incompatible batch dimensions at index %d: a=%d, b=%d, out=%d", i,
                a_dim, b_dim, out_dim);
            assert(0 && "matmul_grad_op: Incompatible batch dimensions!");
        }

        batch_size *= out_dim;
    }

    // Calculate strides for matrix operations (last two dimensions)
    int a_row_stride = a->strides[a->ndim - 2];
    int a_col_stride = a->strides[a->ndim - 1];
    int b_row_stride = b->strides[b->ndim - 2];
    int b_col_stride = b->strides[b->ndim - 1];
    int out_row_stride = out->strides[out->ndim - 2];
    int out_col_stride = out->strides[out->ndim - 1];

    // Compute gradient for tensor a: grad_a += out_grad @ b^T
    if (a->requires_grad)
    {
        if (!a->grad->data->data)
        {
            LOG_ERROR("matmul_grad_op: Tensor 'a' requires grad but its grad is NULL!");
            assert(0 && "matmul_grad_op: Tensor 'a' requires grad but its grad is NULL!");
        }

        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx)
        {
            // Calculate batch offsets with proper broadcasting
            int a_batch_offset = 0;
            int b_batch_offset = 0;
            int out_batch_offset = 0;

            int temp_batch_idx = batch_idx;
            for (int dim = out->ndim - 3; dim >= 0; --dim)
            {
                int a_dim = (dim < a->ndim - 2) ? a->shape[dim] : 1;
                int b_dim = (dim < b->ndim - 2) ? b->shape[dim] : 1;
                int out_dim = (dim < out->ndim - 2) ? out->shape[dim] : 1;

                int coord = temp_batch_idx % out_dim;
                temp_batch_idx /= out_dim;

                if (dim < a->ndim - 2 && a_dim > 1)
                {
                    a_batch_offset += coord * a->strides[dim];
                }
                if (dim < b->ndim - 2 && b_dim > 1)
                {
                    b_batch_offset += coord * b->strides[dim];
                }
                if (dim < out->ndim - 2)
                {
                    out_batch_offset += coord * out->strides[dim];
                }
            }

            // Compute grad_a[i,j] += sum_k(out_grad[i,k] * b[j,k])
            for (int i = 0; i < N; ++i)
            {
                for (int j = 0; j < K; ++j)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < M; ++k)
                    {
                        float out_grad_val =
                            out->grad->data
                                ->data[out_batch_offset + i * out_row_stride + k * out_col_stride];
                        float b_val =
                            b->data->data[b_batch_offset + j * b_row_stride + k * b_col_stride];
                        sum += out_grad_val * b_val;
                    }
                    a->grad->data->data[a_batch_offset + i * a_row_stride + j * a_col_stride] +=
                        sum;
                }
            }
        }
    }

    // Compute gradient for tensor b: grad_b += a^T @ out_grad
    if (b->requires_grad)
    {
        if (!b->grad->data->data)
        {
            LOG_ERROR("matmul_grad_op: Tensor 'b' requires grad but its grad is NULL!");
            assert(0 && "matmul_grad_op: Tensor 'b' requires grad but its grad is NULL!");
        }

        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx)
        {
            // Calculate batch offsets with proper broadcasting
            int a_batch_offset = 0;
            int b_batch_offset = 0;
            int out_batch_offset = 0;

            int temp_batch_idx = batch_idx;
            for (int dim = out->ndim - 3; dim >= 0; --dim)
            {
                int a_dim = (dim < a->ndim - 2) ? a->shape[dim] : 1;
                int b_dim = (dim < b->ndim - 2) ? b->shape[dim] : 1;
                int out_dim = (dim < out->ndim - 2) ? out->shape[dim] : 1;

                int coord = temp_batch_idx % out_dim;
                temp_batch_idx /= out_dim;

                if (dim < a->ndim - 2 && a_dim > 1)
                {
                    a_batch_offset += coord * a->strides[dim];
                }
                if (dim < b->ndim - 2 && b_dim > 1)
                {
                    b_batch_offset += coord * b->strides[dim];
                }
                if (dim < out->ndim - 2)
                {
                    out_batch_offset += coord * out->strides[dim];
                }
            }

            // Compute grad_b[i,j] += sum_k(a[k,i] * out_grad[k,j])
            for (int i = 0; i < K; ++i)
            {
                for (int j = 0; j < M; ++j)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < N; ++k)
                    {
                        float a_val =
                            a->data->data[a_batch_offset + k * a_row_stride + i * a_col_stride];
                        float out_grad_val =
                            out->grad->data
                                ->data[out_batch_offset + k * out_row_stride + j * out_col_stride];
                        sum += a_val * out_grad_val;
                    }
                    b->grad->data->data[b_batch_offset + i * b_row_stride + j * b_col_stride] +=
                        sum;
                }
            }
        }
    }

    LOG_INFO("matmul_grad_op: Exiting function");
}
