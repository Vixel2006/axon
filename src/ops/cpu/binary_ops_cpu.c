#include <immintrin.h>
#include <math.h>
#include <sleef.h>

#include "ops/ops.h"
#include "utils.h"

#define SIMD_WIDTH 8

void add_op(Tensor *a, Tensor *b, Tensor *out) {
  int size = numel(a->shape, a->ndim);
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

      out->data[out_offset] = a->data[a_offset] + b->data[b_offset];
    }
  } else {
    int i = 0;

    for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
      __m256 x = _mm256_loadu_ps(a->data + i);
      __m256 y = _mm256_loadu_ps(b->data + i);
      __m256 z = _mm256_add_ps(x, y);
      _mm256_storeu_ps(out->data + i, z);
    }

    for (; i < size; ++i) {
      out->data[i] = a->data[i] + b->data[i];
    }
  }

  out->requires_grad = a->requires_grad || b->requires_grad ? true : false;
}

void sub_op(Tensor *a, Tensor *b, Tensor *out) {
  int size = numel(a->shape, a->ndim);
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

      out->data[out_offset] = a->data[a_offset] - b->data[b_offset];
    }
  } else {
    int i = 0;

    for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
      __m256 x = _mm256_loadu_ps(a->data + i);
      __m256 y = _mm256_loadu_ps(b->data + i);
      __m256 z = _mm256_sub_ps(x, y);
      _mm256_storeu_ps(out->data + i, z);
    }

    for (; i < size; ++i) {
      out->data[i] = a->data[i] - b->data[i];
    }
  }

  out->requires_grad = a->requires_grad || b->requires_grad ? true : false;
}

void mul_op(Tensor *a, Tensor *b, Tensor *out) {
  int size = numel(a->shape, a->ndim);
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

      out->data[out_offset] = a->data[a_offset] * b->data[b_offset];
    }
  } else {
    int i = 0;

    for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
      __m256 x = _mm256_loadu_ps(a->data + i);
      __m256 y = _mm256_loadu_ps(b->data + i);
      __m256 z = _mm256_mul_ps(x, y);
      _mm256_storeu_ps(out->data + i, z);
    }

    for (; i < size; ++i) {
      out->data[i] = a->data[i] * b->data[i];
    }
  }

  out->requires_grad = a->requires_grad || b->requires_grad ? true : false;
}

void div_op(Tensor *a, Tensor *b, Tensor *out) {
  int size = numel(a->shape, b->ndim);
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

      out->data[out_offset] = a->data[a_offset] / b->data[b_offset];
    }
  } else {
    int i = 0;

    for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
      __m256 x = _mm256_loadu_ps(a->data + i);
      __m256 y = _mm256_loadu_ps(b->data + i);
      __m256 z = _mm256_div_ps(x, y);
      _mm256_storeu_ps(out->data + i, z);
    }

    for (; i < size; ++i) {
      out->data[i] = a->data[i] / b->data[i];
    }
  }

  out->requires_grad = a->requires_grad || b->requires_grad ? true : false;
}

void matmul_op(Tensor *a, Tensor *b, Tensor *out, int N, int K, int P) {
  // 1. Figure out how many "batch matmuls" we need.
  // Example: if a and b are 3D tensors, the leading dim(s) represent batch
  // size. We multiply all leading dims (except the last two, which are the
  // matmul dims).
  int num_batches = 1;
  out->shape = malloc(a->ndim * sizeof(int));
  if (!out->shape) {
    free_tensor(out);
    return;
  }
  out->ndim = a->ndim;
  for (int i = 0; i < out->ndim - 2; ++i) {
    num_batches *= a->shape[i];
    out->shape[i] = a->shape[i];
  }
  out->shape[a->ndim - 2] = N;
  out->shape[a->ndim - 1] = P;
  out->strides = compute_strides(out->shape, out->ndim);
  int size = numel(out->shape, out->ndim);
  out->data = malloc(size * sizeof(float));
  if (!out->data) {
    free_tensor(out);
    return;
  }

  // 2. Precompute per-batch strides so we can jump to the right slice of
  // data.
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
                a->data[a_curr_stride + row + k * a->strides[a->ndim - 1]];
            float b_val =
                b->data[b_curr_stride + col + k * b->strides[b->ndim - 2]];
            sum += a_val * b_val;
          }

          out->data[out_curr_stride + i * out->strides[out->ndim - 2] +
                    j * out->strides[out->ndim - 1]] = sum;
        }
      }
    }
  } else {
    // 3. Vectorization setup.
    // We'll compute dot products in chunks of SIMD_WIDTH=8 floats, and then
    // finish the remainder scalar.
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
                &a->data[a_curr_stride + row + k * a->strides[a->ndim - 1]]);
            __m256 b_vec = _mm256_loadu_ps(
                &b->data[b_curr_stride + col + k * b->strides[b->ndim - 2]]);
            sum_vec =
                _mm256_fmadd_ps(a_vec, b_vec, sum_vec); // a fused multiply-add
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
            sum += a->data[a_curr_stride + row + k * a->strides[a->ndim - 1]] *
                   b->data[b_curr_stride + col + k * b->strides[b->ndim - 2]];
          }

          // Write result into output tensor
          out->data[out_curr_stride + i * out->strides[out->ndim - 2] +
                    j * out->strides[out->ndim - 1]] = sum;
        }
      }
    }
  }

  out->requires_grad = a->requires_grad || b->requires_grad;
}

void conv2d_op(Tensor *in, Tensor *kernel, Tensor *out, const int *kernel_size,
               const int *stride, const int padding) {
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
    out->data[i] = 0.0f;
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
                    float in_val = in->data[in_idx];

                    for (int oc = 0; oc < Cout; ++oc) {
                      int kernel_idx =
                          oc * Cin * Kh * Kw + ic * Kh * Kw + kh * Kw + kw;
                      int out_idx = n * Cout * Hout * Wout + oc * Hout * Wout +
                                    oh * Wout + ow;
                      out->data[out_idx] += in_val * kernel->data[kernel_idx];
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
