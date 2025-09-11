#include "ops/ops.h"
#include "tensor.h"
#include "utils.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>

extern __m256 Sleef_powf8_u10(__m256 x, __m256 y);

#define SIMD_WIDTH 8

void add_op(Tensor *a, Tensor *b, Tensor *out) {
  IDRAK_DEBUG("OP   ", "add_op: Performing element-wise addition\n");

  // Error checking for null tensors
  if (!a || !b || !out) {
    IDRAK_ERROR(
        "add_op ERROR: Input or output tensor is NULL! a=%p, b=%p, out=%p\n",
        (void *)a, (void *)b, (void *)out);
    return;
  }

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
}

void sub_op(Tensor *a, Tensor *b, Tensor *out) {
  IDRAK_DEBUG("OP   ", "sub_op: Performing element-wise subtraction\n");

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
}

void mul_op(Tensor *a, Tensor *b, Tensor *out) {
  IDRAK_DEBUG("OP   ", "mul_op: Performing element-wise multiplication\n");

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
}

void div_op(Tensor *a, Tensor *b, Tensor *out) {
  IDRAK_DEBUG("OP   ", "div_op: Performing element-wise division\n");

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
}

void matmul_op(Tensor *a, Tensor *b, Tensor *out, int N, int K, int P) {
  IDRAK_DEBUG("OP   ",
              "matmul_op: Performing matrix multiplication "
              "(N=%d, K=%d, P=%d)\n",
              N, K, P);

  // Error checking for null tensors
  if (!a || !b || !out) {
    IDRAK_ERROR(
        "matmul_op ERROR: Input or output tensor is NULL! a=%p, b=%p, out=%p\n",
        (void *)a, (void *)b, (void *)out);
    return;
  }

  // Error checking for insufficient dimensions
  if (a->ndim < 2 || b->ndim < 2 || out->ndim < 2) {
    IDRAK_ERROR("matmul_op ERROR: All tensors must have at least 2 dimensions! "
                "a->ndim=%d, b->ndim=%d, out->ndim=%d\n",
                a->ndim, b->ndim, out->ndim);
    return;
  }

  // Error checking for dimension mismatch
  if (a->shape[a->ndim - 1] != K || b->shape[b->ndim - 2] != K) {
    IDRAK_ERROR("matmul_op ERROR: Dimension mismatch! a->shape[last]=%d, "
                "b->shape[second_last]=%d, K=%d\n",
                a->shape[a->ndim - 1], b->shape[b->ndim - 2], K);
    return;
  }

  // Verify output dimensions are correct
  if (out->shape[out->ndim - 2] != N || out->shape[out->ndim - 1] != P) {
    IDRAK_ERROR("matmul_op ERROR: Output tensor dimensions incorrect! "
                "Expected (%d, %d), got (%d, %d)\n",
                N, P, out->shape[out->ndim - 2], out->shape[out->ndim - 1]);
    return;
  }

  // Calculate batch size (should be same for all tensors after broadcasting)
  int batch_size = 1;
  for (int i = 0; i < out->ndim - 2; ++i) {
    batch_size *= out->shape[i];
  }

  // Initialize output to zero
  int size = numel(out->shape, out->ndim);
  for (int i = 0; i < size; ++i) {
    out->data->ptr[i] = 0.0f;
  }

  // Check if we can use SIMD (requires contiguous inner dimensions)
  bool can_use_simd =
      is_contiguous(a) && is_contiguous(b) && is_contiguous(out);

  if (can_use_simd) {
    // Fast path: SIMD vectorized computation
    const int k_simd = (K / SIMD_WIDTH) * SIMD_WIDTH;

    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      // Calculate batch offsets with proper broadcasting
      int a_batch_offset = 0;
      int b_batch_offset = 0;
      int out_batch_offset = 0;

      int temp_batch_idx = batch_idx;
      for (int dim = out->ndim - 3; dim >= 0; --dim) {
        int a_dim = (dim < a->ndim - 2) ? a->shape[dim] : 1;
        int b_dim = (dim < b->ndim - 2) ? b->shape[dim] : 1;
        int out_dim = out->shape[dim];

        int coord = temp_batch_idx % out_dim;
        temp_batch_idx /= out_dim;

        if (dim < a->ndim - 2 && a_dim > 1) {
          a_batch_offset += coord * a->strides[dim];
        }
        if (dim < b->ndim - 2 && b_dim > 1) {
          b_batch_offset += coord * b->strides[dim];
        }
        out_batch_offset += coord * out->strides[dim];
      }

      for (int i = 0; i < N; ++i) {
        int a_row_offset = a_batch_offset + i * a->strides[a->ndim - 2];
        for (int j = 0; j < P; ++j) {
          int b_col_offset = b_batch_offset + j * b->strides[b->ndim - 1];

          // Vectorized dot product
          __m256 sum_vec = _mm256_setzero_ps();

          for (int k = 0; k < k_simd; k += SIMD_WIDTH) {
            __m256 a_vec = _mm256_loadu_ps(
                &a->data->ptr[a_row_offset + k * a->strides[a->ndim - 1]]);
            __m256 b_vec = _mm256_loadu_ps(
                &b->data->ptr[b_col_offset + k * b->strides[b->ndim - 2]]);
            sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
          }

          // Horizontal sum
          __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
          __m128 sum_low = _mm256_castps256_ps128(sum_vec);
          __m128 sum128 = _mm_add_ps(sum_high, sum_low);
          __m128 shuf = _mm_movehdup_ps(sum128);
          __m128 sums = _mm_add_ps(sum128, shuf);
          shuf = _mm_movehl_ps(shuf, sums);
          sums = _mm_add_ss(sums, shuf);
          float sum = _mm_cvtss_f32(sums);

          // Handle remaining elements
          for (int k = k_simd; k < K; ++k) {
            sum += a->data->ptr[a_row_offset + k * a->strides[a->ndim - 1]] *
                   b->data->ptr[b_col_offset + k * b->strides[b->ndim - 2]];
          }

          out->data->ptr[out_batch_offset + i * out->strides[out->ndim - 2] +
                         j * out->strides[out->ndim - 1]] = sum;
        }
      }
    }
  } else {
    // Slow path: respect all strides
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      // Calculate batch offsets with proper broadcasting
      int a_batch_offset = 0;
      int b_batch_offset = 0;
      int out_batch_offset = 0;

      int temp_batch_idx = batch_idx;
      for (int dim = out->ndim - 3; dim >= 0; --dim) {
        int a_dim = (dim < a->ndim - 2) ? a->shape[dim] : 1;
        int b_dim = (dim < b->ndim - 2) ? b->shape[dim] : 1;
        int out_dim = out->shape[dim];

        int coord = temp_batch_idx % out_dim;
        temp_batch_idx /= out_dim;

        if (dim < a->ndim - 2 && a_dim > 1) {
          a_batch_offset += coord * a->strides[dim];
        }
        if (dim < b->ndim - 2 && b_dim > 1) {
          b_batch_offset += coord * b->strides[dim];
        }
        out_batch_offset += coord * out->strides[dim];
      }

      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
          float sum = 0.0f;

          for (int k = 0; k < K; ++k) {
            float a_val =
                a->data->ptr[a_batch_offset + i * a->strides[a->ndim - 2] +
                             k * a->strides[a->ndim - 1]];
            float b_val =
                b->data->ptr[b_batch_offset + k * b->strides[b->ndim - 2] +
                             j * b->strides[b->ndim - 1]];
            sum += a_val * b_val;
          }

          out->data->ptr[out_batch_offset + i * out->strides[out->ndim - 2] +
                         j * out->strides[out->ndim - 1]] = sum;
        }
      }
    }
  }

  IDRAK_DEBUG("OP   ",
              "matmul_op: Matrix multiplication completed successfully\n");
}

void conv2d_op(Tensor *in, Tensor *kernel, Tensor *out, const int *kernel_size,
               const int *stride, const int padding) {
  IDRAK_DEBUG("OP   ", "conv2d_op: Performing 2D convolution\n");

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
  IDRAK_DEBUG("OP   ", "dot_op: Performing dot product\n");

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

        a_offset += coord * a->strides[d];
        b_offset += coord * b->strides[d];
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
}

void pow_op(Tensor *a, Tensor *b, Tensor *out) {
  IDRAK_DEBUG("OP   ", "pow_op: Performing element-wise power\n");

  // Error checking for null tensors
  if (!a || !b || !out) {
    IDRAK_ERROR(
        "pow_op ERROR: Input or output tensor is NULL! a=%p, b=%p, out=%p\n",
        (void *)a, (void *)b, (void *)out);
    return;
  }

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
          powf(a->data->ptr[a_offset], b->data->ptr[b_offset]);
    }
  } else {
    int i = 0;

    for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
      __m256 x = _mm256_loadu_ps(a->data->ptr + i);
      __m256 y = _mm256_loadu_ps(b->data->ptr + i);
      __m256 z = Sleef_powf8_u10(x, y);
      _mm256_storeu_ps(out->data->ptr + i, z);
    }

    for (; i < size; ++i) {
      out->data->ptr[i] = powf(a->data->ptr[i], b->data->ptr[i]);
    }
  }
}
  
