#include "logger.h"
#include "ops/binary_ops.h"
#include "ops/init_ops.h"
#include "tensor.h"
#include "utils.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>
#include <string.h> // For memset

extern __m256 Sleef_powf8_u10(__m256 x, __m256 y);

#define SIMD_WIDTH 8

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

static inline bool check_tensors(Tensor* a, Tensor* b, Tensor* out, const char* op_name)
{
    if (!a || !b || !out)
    {
        LOG_ERROR("%s ERROR: NULL tensor! a=%p, b=%p, out=%p", op_name, (void*) a, (void*) b,
                  (void*) out);
        return false;
    }
    return true;
}

static inline bool check_tensors_unary_or_dot(Tensor* a, Tensor* out, const char* op_name)
{
    if (!a || !out)
    {
        LOG_ERROR("%s ERROR: NULL tensor! a=%p, out=%p", op_name, (void*) a, (void*) out);
        return false;
    }
    return true;
}

static inline float* alloc_tensor_data(int size, const char* op_name)
{
    float* data = malloc(sizeof(float) * size);
    if (!data)
    {
        LOG_ERROR("%s ERROR: Failed to allocate memory for %d floats", op_name, size);
        return NULL;
    }
    return data;
}

static inline bool can_use_simd(Tensor* a, Tensor* b, Tensor* out)
{
    return is_contiguous(a) && is_contiguous(b) && is_contiguous(out);
}

void add_op_cpu(Tensor* a, Tensor* b, Tensor* out)
{
    LOG_INFO("OP: add_op: Performing element-wise addition");
    LOG_INFO("Tensor Pointers - a: data=%p, grad=%p | b: data=%p, grad=%p | out: data=%p, grad=%p",
             (void*) a->data->data, (void*) a->grad->data, (void*) b->data->data,
             (void*) b->grad->data, (void*) out->data->data, (void*) out->grad->data);

    if (!check_tensors(a, b, out, "add_op")) return;

    int size = numel(out->shape, out->ndim);
    float* data = alloc_tensor_data(size, "add_op");
    if (!data) return;

    if (!can_use_simd(a, b, out))
    {
        int a_offset, b_offset, out_offset;
        for (int linear = 0; linear < size; ++linear)
        {
            COMPUTE_OFFSETS(linear, out, a->strides, b->strides, out->strides, a_offset, b_offset,
                            out_offset);
            data[out_offset] = a->data->data[a_offset] + b->data->data[b_offset];
        }
    }
    else
    {
        int i = 0;
        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
        {
            __m256 x = _mm256_loadu_ps(a->data->data + i);
            __m256 y = _mm256_loadu_ps(b->data->data + i);
            __m256 z = _mm256_add_ps(x, y);
            _mm256_storeu_ps(data + i, z);
        }
        for (; i < size; ++i)
        {
            data[i] = a->data->data[i] + b->data->data[i];
        }
    }
    from_data(out, data);
    SAFE_FREE(&data, free);
}

void sub_op_cpu(Tensor* a, Tensor* b, Tensor* out)
{
    LOG_INFO("OP: sub_op: Performing element-wise subtraction");
    LOG_INFO("Tensor Pointers - a: data=%p, grad=%p | b: data=%p, grad=%p | out: data=%p, grad=%p",
             (void*) a->data->data, (void*) a->grad->data, (void*) b->data->data,
             (void*) b->grad->data, (void*) out->data->data, (void*) out->grad->data);

    if (!check_tensors(a, b, out, "sub_op")) return;

    int size = numel(out->shape, out->ndim);
    float* data = alloc_tensor_data(size, "sub_op");
    if (!data) return;

    if (!can_use_simd(a, b, out))
    {
        int a_offset, b_offset, out_offset;
        for (int linear = 0; linear < size; ++linear)
        {
            COMPUTE_OFFSETS(linear, out, a->strides, b->strides, out->strides, a_offset, b_offset,
                            out_offset);
            data[out_offset] = a->data->data[a_offset] - b->data->data[b_offset];
        }
    }
    else
    {
        int i = 0;
        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
        {
            __m256 x = _mm256_loadu_ps(a->data->data + i);
            __m256 y = _mm256_loadu_ps(b->data->data + i);
            __m256 z = _mm256_sub_ps(x, y);
            _mm256_storeu_ps(data + i, z);
        }
        for (; i < size; ++i)
        {
            data[i] = a->data->data[i] - b->data->data[i];
        }
    }

    from_data(out, data);
    SAFE_FREE(&data, free);
}

void mul_op_cpu(Tensor* a, Tensor* b, Tensor* out)
{
    LOG_INFO("OP: mul_op: Performing element-wise multiplication");
    LOG_INFO("Tensor Pointers - a: data=%p, grad=%p | b: data=%p, grad=%p | out: data=%p, grad=%p",
             (void*) a->data->data, (void*) a->grad->data, (void*) b->data->data,
             (void*) b->grad->data, (void*) out->data->data, (void*) out->grad->data);

    if (!check_tensors(a, b, out, "mul_op")) return;

    int size = numel(out->shape, out->ndim);
    float* data = alloc_tensor_data(size, "mul_op");
    if (!data) return;

    if (!can_use_simd(a, b, out))
    {
        int a_offset, b_offset, out_offset;
        for (int linear = 0; linear < size; ++linear)
        {
            COMPUTE_OFFSETS(linear, out, a->strides, b->strides, out->strides, a_offset, b_offset,
                            out_offset);
            data[out_offset] = a->data->data[a_offset] * b->data->data[b_offset];
        }
    }
    else
    {
        int i = 0;
        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
        {
            __m256 x = _mm256_loadu_ps(a->data->data + i);
            __m256 y = _mm256_loadu_ps(b->data->data + i);
            __m256 z = _mm256_mul_ps(x, y);
            _mm256_storeu_ps(data + i, z);
        }
        for (; i < size; ++i)
        {
            data[i] = a->data->data[i] * b->data->data[i];
        }
    }
    from_data(out, data);
    SAFE_FREE(&data, free);
}

void div_op_cpu(Tensor* a, Tensor* b, Tensor* out)
{
    LOG_INFO("OP: div_op: Performing element-wise division");
    LOG_INFO("Tensor Pointers - a: data=%p, grad=%p | b: data=%p, grad=%p | out: data=%p, grad=%p",
             (void*) a->data->data, (void*) a->grad->data, (void*) b->data->data,
             (void*) b->grad->data, (void*) out->data->data, (void*) out->grad->data);

    if (!check_tensors(a, b, out, "div_op")) return;

    int size = numel(out->shape, out->ndim);
    float* data = alloc_tensor_data(size, "div_op");
    if (!data) return;

    if (!can_use_simd(a, b, out))
    {
        int a_offset, b_offset, out_offset;
        for (int linear = 0; linear < size; ++linear)
        {
            COMPUTE_OFFSETS(linear, out, a->strides, b->strides, out->strides, a_offset, b_offset,
                            out_offset);
            data[out_offset] = a->data->data[a_offset] / b->data->data[b_offset];
        }
    }
    else
    {
        int i = 0;
        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
        {
            __m256 x = _mm256_loadu_ps(a->data->data + i);
            __m256 y = _mm256_loadu_ps(b->data->data + i);
            __m256 z = _mm256_div_ps(x, y);
            _mm256_storeu_ps(data + i, z);
        }
        for (; i < size; ++i)
        {
            data[i] = a->data->data[i] / b->data->data[i];
        }
    }
    from_data(out, data);
    SAFE_FREE(&data, free);
}

void matmul_op_cpu(Tensor* a, Tensor* b, Tensor* out, int N, int K, int P)
{
    LOG_INFO("OP: matmul_op: Performing matrix multiplication (N=%d, K=%d, P=%d)", N, K, P);
    LOG_INFO("Tensor Pointers - a: data=%p, grad=%p | b: data=%p, grad=%p | out: data=%p, grad=%p",
             (void*) a->data->data, (void*) a->grad->data, (void*) b->data->data,
             (void*) b->grad->data, (void*) out->data->data, (void*) out->grad->data);

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

    LOG_INFO("OP: matmul_op: Matrix multiplication completed successfully");
}

void conv2d_op_cpu(Tensor* in, Tensor* kernel, Tensor* out, const int* kernel_size,
                   const int* stride, const int padding)
{
    LOG_INFO("OP: conv2d_op: Performing 2D convolution");
    LOG_INFO(
        "Tensor Pointers - in: data=%p, grad=%p | kernel: data=%p, grad=%p | out: data=%p, grad=%p",
        (void*) in->data->data, (void*) in->grad->data, (void*) kernel->data->data,
        (void*) kernel->grad->data, (void*) out->data->data, (void*) out->grad->data);

    if (!check_tensors(in, kernel, out, "conv2d_op")) return;

    int Cin = kernel_size[0];
    int Cout = kernel_size[1];
    int Kh = kernel_size[2];
    int Kw = kernel_size[3];
    int Sh = stride[0];
    int Sw = stride[1];
    int H = in->shape[in->ndim - 2];
    int W = in->shape[in->ndim - 1];
    int Hout = (H + 2 * padding - Kh) / Sh + 1;
    int Wout = (W + 2 * padding - Kw) / Sw + 1;
    int N = in->shape[0];

    int out_size = N * Cout * Hout * Wout;
    float* data = alloc_tensor_data(out_size, "conv2d_op");
    if (!data) return;
    memset(data, 0, sizeof(float) * out_size);

    const int TILE_H = 16;
    const int TILE_W = 16;

    for (int n = 0; n < N; ++n)
    {
        for (int ic = 0; ic < Cin; ++ic)
        {
            for (int kh = 0; kh < Kh; ++kh)
            {
                for (int kw = 0; kw < Kw; ++kw)
                {
                    for (int oh_start = 0; oh_start < Hout; oh_start += TILE_H)
                    {
                        int oh_end = (oh_start + TILE_H > Hout) ? Hout : oh_start + TILE_H;

                        for (int ow_start = 0; ow_start < Wout; ow_start += TILE_W)
                        {
                            int ow_end = (ow_start + TILE_W > Wout) ? Wout : ow_start + TILE_W;

                            for (int oh = oh_start; oh < oh_end; ++oh)
                            {
                                for (int ow = ow_start; ow < ow_end; ++ow)
                                {
                                    int ih = oh * Sh - padding + kh;
                                    int iw = ow * Sw - padding + kw;

                                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                    {
                                        int in_idx = n * Cin * H * W + ic * H * W + ih * W + iw;
                                        float in_val = in->data->data[in_idx];

                                        for (int oc = 0; oc < Cout; ++oc)
                                        {
                                            int kernel_idx =
                                                oc * Cin * Kh * Kw + ic * Kh * Kw + kh * Kw + kw;
                                            int out_idx = n * Cout * Hout * Wout +
                                                          oc * Hout * Wout + oh * Wout + ow;
                                            data[out_idx] +=
                                                in_val * kernel->data->data[kernel_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    from_data(out, data);
    SAFE_FREE(&data, free);
}

void dot_op_cpu(Tensor* a, Tensor* b, Tensor* out)
{
    LOG_INFO("OP: dot_op: Performing dot product");
    LOG_INFO("Tensor Pointers - a: data=%p, grad=%p | b: data=%p, grad=%p | out: data=%p, grad=%p",
             (void*) a->data->data, (void*) a->grad->data, (void*) b->data->data,
             (void*) b->grad->data, (void*) out->data->data, (void*) out->grad->data);

    if (!check_tensors(a, b, out, "dot_op")) return;

    int size = numel(a->shape, a->ndim);
    float* data = alloc_tensor_data(1, "dot_op");
    if (!data) return;

    if (!can_use_simd(a, b, out))
    {
        float sum = 0.0f;
        int a_offset, b_offset, dummy_out_offset;
        for (int linear = 0; linear < size; ++linear)
        {
            COMPUTE_OFFSETS(linear, a, a->strides, b->strides, a->strides, a_offset, b_offset,
                            dummy_out_offset);
            sum += a->data->data[a_offset] * b->data->data[b_offset];
        }
        data[0] = sum;
    }
    else
    {
        float sum = 0.0f;
        int i = 0;

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
        {
            __m256 x = _mm256_loadu_ps(a->data->data + i);
            __m256 y = _mm256_loadu_ps(b->data->data + i);
            __m256 prod_vec = _mm256_mul_ps(x, y);

            __m128 sum_high = _mm256_extractf128_ps(prod_vec, 1);
            __m128 sum_low = _mm256_castps256_ps128(prod_vec);
            __m128 sum128 = _mm_add_ps(sum_high, sum_low);
            __m128 shuf = _mm_movehdup_ps(sum128);
            __m128 sums = _mm_add_ps(sum128, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            sum += _mm_cvtss_f32(sums);
        }

        for (; i < size; ++i)
        {
            sum += a->data->data[i] * b->data->data[i];
        }
        data[0] = sum;
    }
    from_data(out, data);
    SAFE_FREE(&data, free);
}

void pow_op_cpu(Tensor* a, Tensor* b, Tensor* out)
{
    LOG_INFO("OP: pow_op: Performing element-wise power");
    LOG_INFO("Tensor Pointers - a: data=%p, grad=%p | b: data=%p, grad=%p | out: data=%p, grad=%p",
             (void*) a->data->data, (void*) a->grad->data, (void*) b->data->data,
             (void*) b->grad->data, (void*) out->data->data, (void*) out->grad->data);

    if (!check_tensors(a, b, out, "pow_op")) return;

    int size = numel(out->shape, out->ndim);
    float* data = alloc_tensor_data(size, "pow_op");
    if (!data) return;

    if (!can_use_simd(a, b, out))
    {
        int a_offset, b_offset, out_offset;
        for (int linear = 0; linear < size; ++linear)
        {
            COMPUTE_OFFSETS(linear, out, a->strides, b->strides, out->strides, a_offset, b_offset,
                            out_offset);
            data[out_offset] = powf(a->data->data[a_offset], b->data->data[b_offset]);
        }
    }
    else
    {
        int i = 0;
        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH)
        {
            __m256 x = _mm256_loadu_ps(a->data->data + i);
            __m256 y = _mm256_loadu_ps(b->data->data + i);
            __m256 z = Sleef_powf8_u10(x, y);
            _mm256_storeu_ps(data + i, z);
        }
        for (; i < size; ++i)
        {
            data[i] = powf(a->data->data[i], b->data->data[i]);
        }
    }
    from_data(out, data);
    SAFE_FREE(&data, free);
}
