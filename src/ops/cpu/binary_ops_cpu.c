#include "ops/ops.h"
#include "tensor.h"
#include "utils.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>

#define SIMD_WIDTH 8

void add_op(Tensor *a, Tensor *b, Tensor *out) {
  DEBUG_PRINT("[IDRAK_DEBUG] add_op: Performing element-wise addition\n");

  int size = numel(out->shape, out->ndim);

  if (!is_contiguous(a) || !is_contiguous(b) || !is_contiguous(out)) {
    int ndim = out->ndim;
    int *shape = out->shape;

    int *a_strides = a->strides;
    int *b_strides = b->strides;
    int *out_strides = out->strides;

    for (int linear = 0; linear < size; ++linear) {
      int idx = linear;
      int a_offset = 0, b_offset = 0, out_offset = 0;

      for (int d = ndim - 1; d >= 0; --d) {
        int coord = idx % shape[d];
        idx /= shape[d];

        a_offset += coord * a_strides[d];
        b_offset += coord * b_strides[d];
        out_offset += coord * out_strides[d];
      }

      out->data->ptr[out_offset] =
          a->data->ptr[a_offset] + b->data->ptr[b_offset];
    }
  } else {
    int i = 0;

    for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
      __m256 x = _mm256_loadu_ps(a->data->ptr + i);
      __m256 y = _mm256_loadu_ps(b->data->ptr + i);
      __m256 z = _mm256_add_ps(x, y);
      _mm256_storeu_ps(out->data->ptr + i, z);
    }

    for (; i < size; ++i) {
      out->data->ptr[i] = a->data->ptr[i] + b->data->ptr[i];
    }
  }

  out->requires_grad = a->requires_grad || b->requires_grad ? true : false;
}

void sub_op(Tensor *a, Tensor *b, Tensor *out) {
  DEBUG_PRINT("[IDRAK_DEBUG] sub_op: Performing element-wise subtraction\n");

  int size = numel(out->shape, out->ndim);

  if (!is_contiguous(a) || !is_contiguous(b) || !is_contiguous(out)) {
    int ndim = out->ndim;
    int *shape = out->shape;

    int *a_strides = a->strides;
    int *b_strides = b->strides;
    int *out_strides = out->strides;

    for (int linear = 0; linear < size; ++linear) {
      int idx = linear;
      int a_offset = 0, b_offset = 0, out_offset = 0;

      for (int d = ndim - 1; d >= 0; --d) {
        int coord = idx % shape[d];
        idx /= shape[d];

        a_offset += coord * a_strides[d];
        b_offset += coord * b_strides[d];
        out_offset += coord * out_strides[d];
      }

      out->data->ptr[out_offset] =
          a->data->ptr[a_offset] - b->data->ptr[b_offset];
    }
  } else {
    int i = 0;

    for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
      __m256 x = _mm256_loadu_ps(a->data->ptr + i);
      __m256 y = _mm256_loadu_ps(b->data->ptr + i);
      __m256 z = _mm256_sub_ps(x, y);
      _mm256_storeu_ps(out->data->ptr + i, z);
    }

    for (; i < size; ++i) {
      out->data->ptr[i] = a->data->ptr[i] - b->data->ptr[i];
    }
  }

  out->requires_grad = a->requires_grad || b->requires_grad ? true : false;
}

void mul_op(Tensor *a, Tensor *b, Tensor *out) {
  DEBUG_PRINT("[IDRAK_DEBUG] mul_op: Performing element-wise multiplication\n");

  int size = numel(out->shape, out->ndim);

  if (!is_contiguous(a) || !is_contiguous(b) || !is_contiguous(out)) {
    int ndim = out->ndim;
    int *shape = out->shape;

    int *a_strides = a->strides;
    int *b_strides = b->strides;
    int *out_strides = out->strides;

    for (int linear = 0; linear < size; ++linear) {
      int idx = linear;
      int a_offset = 0, b_offset = 0, out_offset = 0;

      for (int d = ndim - 1; d >= 0; --d) {
        int coord = idx % shape[d];
        idx /= shape[d];

        a_offset += coord * a_strides[d];
        b_offset += coord * b_strides[d];
        out_offset += coord * out_strides[d];
      }

      out->data->ptr[out_offset] =
          a->data->ptr[a_offset] * b->data->ptr[b_offset];
    }
  } else {
    int i = 0;

    for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
      __m256 x = _mm256_loadu_ps(a->data->ptr + i);
      __m256 y = _mm256_loadu_ps(b->data->ptr + i);
      __m256 z = _mm256_mul_ps(x, y);
      _mm256_storeu_ps(out->data->ptr + i, z);
    }

    for (; i < size; ++i) {
      out->data->ptr[i] = a->data->ptr[i] * b->data->ptr[i];
    }
  }

  out->requires_grad = a->requires_grad || b->requires_grad ? true : false;
}

void div_op(Tensor *a, Tensor *b, Tensor *out) {
  DEBUG_PRINT("[IDRAK_DEBUG] div_op: Performing element-wise division\n");

  int size = numel(out->shape, out->ndim);

  if (!is_contiguous(a) || !is_contiguous(b) || !is_contiguous(out)) {
    int ndim = out->ndim;
    int *shape = out->shape;

    int *a_strides = a->strides;
    int *b_strides = b->strides;
    int *out_strides = out->strides;

    for (int linear = 0; linear < size; ++linear) {
      int idx = linear;
      int a_offset = 0, b_offset = 0, out_offset = 0;

      for (int d = ndim - 1; d >= 0; --d) {
        int coord = idx % shape[d];
        idx /= shape[d];

        a_offset += coord * a_strides[d];
        b_offset += coord * b_strides[d];
        out_offset += coord * out_strides[d];
      }

      out->data->ptr[out_offset] =
          a->data->ptr[a_offset] / b->data->ptr[b_offset];
    }
  } else {
    int i = 0;

    for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
      __m256 x = _mm256_loadu_ps(a->data->ptr + i);
      __m256 y = _mm256_loadu_ps(b->data->ptr + i);
      __m256 z = _mm256_div_ps(x, y);
      _mm256_storeu_ps(out->data->ptr + i, z);
    }

    for (; i < size; ++i) {
      out->data->ptr[i] = a->data->ptr[i] / b->data->ptr[i];
    }
  }

  out->requires_grad = a->requires_grad || b->requires_grad ? true : false;
}

void matmul_op(Tensor *a, Tensor *b, Tensor *out, int N, int K, int P) {
  DEBUG_PRINT("[IDRAK_DEBUG] matmul_op: Performing matrix multiplication "
              "(N=%d, K=%d, P=%d)\n",
              N, K, P);

  // 1. Figure out how many "batch matmuls" we need.
  int num_batches = 1;

  // Set up output tensor dimensions
  out->ndim = a->ndim;
  for (int i = 0; i < out->ndim - 2; ++i) {
    num_batches *= a->shape[i];
    out->shape[i] =
        a->shape[i]; // This modifies existing allocation, which is fine
  }
  out->shape[a->ndim - 2] = N;
  out->shape[a->ndim - 1] = P;

  // Free old strides and compute new ones
  if (out->strides) {
    free(out->strides);
  }
  out->strides = compute_strides(out->shape, out->ndim);

  int size = numel(out->shape, out->ndim);

  // Check if we need to resize the data buffer
  // (In a real implementation, you might want to check the current size)
  // For now, we'll assume the output tensor was allocated with the correct size
  // and just zero out the existing data
  for (int i = 0; i < size; ++i) {
    out->data->ptr[i] = 0.0f;
  }

  // 2. Precompute per-batch strides so we can jump to the right slice of data.
  int a_batch_stride = (a->ndim > 2) ? a->strides[a->ndim - 3] : N * K;
  int b_batch_stride = (b->ndim > 2) ? b->strides[b->ndim - 3] : K * P;
  int out_batch_stride = (out->ndim > 2) ? out->strides[out->ndim - 3] : N * P;

  if (!is_contiguous(a) || !is_contiguous(b) || !is_contiguous(out)) {
    // Slow path: pure scalar, respect strides
    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      int a_curr_stride = batch_idx * a_batch_stride;
      int b_curr_stride = batch_idx * b_batch_stride;
      int out_curr_stride = batch_idx * out_batch_stride;

      for (int i = 0; i < N; ++i) {
        int row = i * a->strides[a->ndim - 2];
        for (int j = 0; j < P; ++j) {
          int col = j * b->strides[b->ndim - 1];
          float sum = 0.0f;

          for (int k = 0; k < K; ++k) {
            float a_val =
                a->data->ptr[a_curr_stride + row + k * a->strides[a->ndim - 1]];
            float b_val =
                b->data->ptr[b_curr_stride + col + k * b->strides[b->ndim - 2]];
            sum += a_val * b_val;
          }

          out->data->ptr[out_curr_stride + i * out->strides[out->ndim - 2] +
                         j * out->strides[out->ndim - 1]] = sum;
        }
      }
    }
  } else {
    // 3. Vectorization setup.
    const int k_simd = (K / SIMD_WIDTH) * SIMD_WIDTH;

    // 4. Batched matmul loop: [num_batches, N, P].
    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      int a_curr_stride = batch_idx * a_batch_stride;
      int b_curr_stride = batch_idx * b_batch_stride;
      int out_curr_stride = batch_idx * out_batch_stride;

      for (int i = 0; i < N; ++i) {
        int row = i * a->strides[a->ndim - 2];
        for (int j = 0; j < P; ++j) {
          int col = j * b->strides[b->ndim - 1];

          // Vectorized dot product over the K dimension
          __m256 sum_vec = _mm256_setzero_ps();

          for (int k = 0; k < k_simd; k += SIMD_WIDTH) {
            __m256 a_vec = _mm256_loadu_ps(
                &a->data
                     ->ptr[a_curr_stride + row + k * a->strides[a->ndim - 1]]);
            __m256 b_vec = _mm256_loadu_ps(
                &b->data
                     ->ptr[b_curr_stride + col + k * b->strides[b->ndim - 2]]);
            sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
          }

          // Horizontal sum across the SIMD vector
          __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
          __m128 sum_low = _mm256_castps256_ps128(sum_vec);
          __m128 sum128 = _mm_add_ps(sum_high, sum_low);
          __m128 shuf = _mm_movehdup_ps(sum128);
          __m128 sums = _mm_add_ps(sum128, shuf);
          shuf = _mm_movehl_ps(shuf, sums);
          sums = _mm_add_ss(sums, shuf);
          float sum = _mm_cvtss_f32(sums);

          // Finish leftover elements if K is not a multiple of 8
          for (int k = k_simd; k < K; ++k) {
            sum +=
                a->data
                    ->ptr[a_curr_stride + row + k * a->strides[a->ndim - 1]] *
                b->data->ptr[b_curr_stride + col + k * b->strides[b->ndim - 2]];
          }

          // Write result into output tensor
          out->data->ptr[out_curr_stride + i * out->strides[out->ndim - 2] +
                         j * out->strides[out->ndim - 1]] = sum;
        }
      }
    }
  }

  out->requires_grad = a->requires_grad || b->requires_grad;
}

void conv2d_op(Tensor *in, Tensor *kernel, Tensor *out, const int *kernel_size,
               const int *stride, const int padding) {
  DEBUG_PRINT("[IDRAK_DEBUG] conv2d_op: Performing 2D convolution\n");

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

  // Initialize output to zero
  int out_size = N * Cout * Hout * Wout;
  for (int i = 0; i < out_size; ++i) {
    out->data->ptr[i] = 0.0f;
  }

  // Tile sizes
  const int TILE_H = 16;
  const int TILE_W = 16;

  for (int n = 0; n < N; ++n) {
    for (int ic = 0; ic < Cin; ++ic) {
      for (int kh = 0; kh < Kh; ++kh) {
        for (int kw = 0; kw < Kw; ++kw) {
          for (int oh_start = 0; oh_start < Hout; oh_start += TILE_H) {
            int oh_end = (oh_start + TILE_H > Hout) ? Hout : oh_start + TILE_H;

            for (int ow_start = 0; ow_start < Wout; ow_start += TILE_W) {
              int ow_end =
                  (ow_start + TILE_W > Wout) ? Wout : ow_start + TILE_W;

              for (int oh = oh_start; oh < oh_end; ++oh) {
                for (int ow = ow_start; ow < ow_end; ++ow) {
                  int ih = oh * Sh - padding + kh;
                  int iw = ow * Sw - padding + kw;

                  if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    int in_idx = n * Cin * H * W + ic * H * W + ih * W + iw;
                    float in_val = in->data->ptr[in_idx];

                    for (int oc = 0; oc < Cout; ++oc) {
                      int kernel_idx =
                          oc * Cin * Kh * Kw + ic * Kh * Kw + kh * Kw + kw;
                      int out_idx = n * Cout * Hout * Wout + oc * Hout * Wout +
                                    oh * Wout + ow;
                      out->data->ptr[out_idx] +=
                          in_val * kernel->data->ptr[kernel_idx];
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
}

void dot_op(Tensor *a, Tensor *b, Tensor *out) {
  DEBUG_PRINT("[IDRAK_DEBUG] dot_op: Performing dot product\n");

  int size = numel(a->shape, a->ndim);

  if (!is_contiguous(a) || !is_contiguous(b)) {
    float sum = 0.0f;
    int *a_strides = a->strides;
    int *b_strides = b->strides;

    for (int linear = 0; linear < size; ++linear) {
      int idx = linear;
      int a_offset = 0, b_offset = 0;

      for (int d = a->ndim - 1; d >= 0; --d) {
        int coord = idx % a->shape[d];
        idx /= a->shape[d];

        a_offset += coord * a_strides[d];
        b_offset += coord * b_strides[d];
      }
      sum += a->data->ptr[a_offset] * b->data->ptr[b_offset];
    }
    out->data->ptr[0] = sum;
  } else {
    float sum = 0.0f;
    int i = 0;

    for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
      __m256 x = _mm256_loadu_ps(a->data->ptr + i);
      __m256 y = _mm256_loadu_ps(b->data->ptr + i);
      __m256 prod_vec = _mm256_mul_ps(x, y);

      // Horizontal sum across the SIMD vector
      __m128 sum_high = _mm256_extractf128_ps(prod_vec, 1);
      __m128 sum_low = _mm256_castps256_ps128(prod_vec);
      __m128 sum128 = _mm_add_ps(sum_high, sum_low);
      __m128 shuf = _mm_movehdup_ps(sum128);
      __m128 sums = _mm_add_ps(sum128, shuf);
      shuf = _mm_movehl_ps(shuf, sums);
      sums = _mm_add_ss(sums, shuf);
      sum += _mm_cvtss_f32(sums);
    }

    for (; i < size; ++i) {
      sum += a->data->ptr[i] * b->data->ptr[i];
    }
    out->data->ptr[0] = sum;
  }

  out->requires_grad = a->requires_grad || b->requires_grad ? true : false;
}
