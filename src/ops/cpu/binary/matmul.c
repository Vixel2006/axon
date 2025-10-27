#include "ops/cpu/binary.h"

void matmul_op_cpu(Tensor* a, Tensor* b, Tensor* out, int N, int K, int P)
{
    LOG_INFO("matmul_op_cpu: Entering function with N=%d, K=%d, P=%d", N, K, P);

    if (!check_tensors(a, b, out, "matmul_op")) return;

    if (a->ndim < 2 || b->ndim < 2 || out->ndim < 2)
    {
        LOG_ERROR("matmul_op ERROR: All tensors must have at least 2 dimensions! "
                  "a->ndim=%d, b->ndim=%d, out->ndim=%d",
                  a->ndim, b->ndim, out->ndim);
        return;
    }

    if (a->shape[a->ndim - 1] != K || b->shape[b->ndim - 2] != K)
    {
        LOG_ERROR("matmul_op ERROR: Dimension mismatch! a->shape[last]=%d, "
                  "b->shape[second_last]=%d, K=%d",
                  a->shape[a->ndim - 1], b->shape[b->ndim - 2], K);
        return;
    }

    if (out->shape[out->ndim - 2] != N || out->shape[out->ndim - 1] != P)
    {
        LOG_ERROR("matmul_op ERROR: Output tensor dimensions incorrect! "
                  "Expected (%d, %d), got (%d, %d)",
                  N, P, out->shape[out->ndim - 2], out->shape[out->ndim - 1]);
        return;
    }

    int batch_size = 1;
    for (int i = 0; i < out->ndim - 2; ++i)
    {
        batch_size *= out->shape[i];
    }

    int size = numel(out->shape, out->ndim);
    float* data = alloc_tensor_data(size, "matmul_op");
    if (!data) return;
    memset(data, 0, sizeof(float) * size);

    bool use_simd_path = can_use_simd(a, b, out);

    if (use_simd_path)
    {
        const int k_simd = (K / SIMD_WIDTH) * SIMD_WIDTH;

        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx)
        {
            int a_batch_offset = 0;
            int b_batch_offset = 0;
            int out_batch_offset = 0;

            int temp_batch_idx = batch_idx;
            for (int dim = out->ndim - 3; dim >= 0; --dim)
            {
                int a_dim = (dim < a->ndim - 2) ? a->shape[dim] : 1;
                int b_dim = (dim < b->ndim - 2) ? b->shape[dim] : 1;
                int out_dim = out->shape[dim];

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
                out_batch_offset += coord * out->strides[dim];
            }

            for (int i = 0; i < N; ++i)
            {
                int a_row_offset = a_batch_offset + i * a->strides[a->ndim - 2];
                for (int j = 0; j < P; ++j)
                {
                    int b_col_offset = b_batch_offset + j * b->strides[b->ndim - 1];

                    __m256 sum_vec = _mm256_setzero_ps();

                    for (int k = 0; k < k_simd; k += SIMD_WIDTH)
                    {
                        __m256 a_vec = _mm256_loadu_ps(a->data->data + a_row_offset +
                                                       k * a->strides[a->ndim - 1]);
                        __m256 b_vec = _mm256_loadu_ps(b->data->data + b_col_offset +
                                                       k * b->strides[b->ndim - 2]);
                        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
                    }

                    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
                    __m128 sum_low = _mm256_castps256_ps128(sum_vec);
                    __m128 sum128 = _mm_add_ps(sum_high, sum_low);
                    __m128 shuf = _mm_movehdup_ps(sum128);
                    __m128 sums = _mm_add_ps(sum128, shuf);
                    shuf = _mm_movehl_ps(shuf, sums);
                    sums = _mm_add_ss(sums, shuf);
                    float sum = _mm_cvtss_f32(sums);

                    for (int k = k_simd; k < K; ++k)
                    {
                        sum += a->data->data[a_row_offset + k * a->strides[a->ndim - 1]] *
                               b->data->data[b_col_offset + k * b->strides[b->ndim - 2]];
                    }

                    data[out_batch_offset + i * out->strides[out->ndim - 2] +
                         j * out->strides[out->ndim - 1]] = sum;
                }
            }
        }
    }
    else
    {
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx)
        {
            int a_batch_offset = 0;
            int b_batch_offset = 0;
            int out_batch_offset = 0;

            int temp_batch_idx = batch_idx;
            for (int dim = out->ndim - 3; dim >= 0; --dim)
            {
                int a_dim = (dim < a->ndim - 2) ? a->shape[dim] : 1;
                int b_dim = (dim < b->ndim - 2) ? b->shape[dim] : 1;
                int out_dim = out->shape[dim];

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
                out_batch_offset += coord * out->strides[dim];
            }

            for (int i = 0; i < N; ++i)
            {
                for (int j = 0; j < P; ++j)
                {
                    float sum = 0.0f;

                    for (int k = 0; k < K; ++k)
                    {
                        float a_val = a->data->data[a_batch_offset + i * a->strides[a->ndim - 2] +
                                                    k * a->strides[a->ndim - 1]];
                        float b_val = b->data->data[b_batch_offset + k * b->strides[b->ndim - 2] +
                                                    j * b->strides[b->ndim - 1]];
                        sum += a_val * b_val;
                    }

                    data[out_batch_offset + i * out->strides[out->ndim - 2] +
                         j * out->strides[out->ndim - 1]] = sum;
                }
            }
        }
    }
    from_data(out, data);
    if (!out->data)
    {
        LOG_ERROR("matmul_op ERROR: Failed to set output tensor data.");
        return;
    }
    SAFE_FREE(&data, free);
}
