#include "autograd.h"
#include "ops.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>

void add_op(Tensor *a, Tensor *b, Tensor *out) {
  int i = 0;
  int size = numel(a->shape, a->ndim);

  for (; i + 7 < size; i += 8) {
    __m256 x = _mm256_loadu_ps(a->data + i);
    __m256 y = _mm256_loadu_ps(b->data + i);
    __m256 z = _mm256_add_ps(x, y);
    _mm256_storeu_ps(out->data + i, z);
  }

  for (; i < size; ++i) {
    out->data[i] = a->data[i] + b->data[i];
  }

  out->requires_grad = a->requires_grad || b->requires_grad ? true : false;

  if (!out->requires_grad) {
    free_tensor(a);
    free_tensor(b);
  }
}

void sub_op(Tensor *a, Tensor *b, Tensor *out) {
  int i = 0;
  int size = numel(a->shape, a->ndim);

  for (; i + 7 < size; i += 8) {
    __m256 x = _mm256_loadu_ps(a->data + i);
    __m256 y = _mm256_loadu_ps(b->data + i);
    __m256 z = _mm256_sub_ps(x, y);
    _mm256_storeu_ps(out->data + i, z);
  }

  for (; i < size; ++i) {
    out->data[i] = a->data[i] - b->data[i];
  }

  out->requires_grad = a->requires_grad || b->requires_grad ? true : false;

  if (!out->requires_grad) {
    free_tensor(a);
    free_tensor(b);
  }
}

void mul_op(Tensor *a, Tensor *b, Tensor *out) {
  int i = 0;
  int size = numel(a->shape, a->ndim);

  for (; i + 7 < size; i += 8) {
    __m256 x = _mm256_loadu_ps(a->data + i);
    __m256 y = _mm256_loadu_ps(b->data + i);
    __m256 z = _mm256_mul_ps(x, y);
    _mm256_storeu_ps(out->data + i, z);
  }

  for (; i < size; ++i) {
    out->data[i] = a->data[i] * b->data[i];
  }

  out->requires_grad = a->requires_grad || b->requires_grad ? true : false;

  if (!out->requires_grad) {
    free_tensor(a);
    free_tensor(b);
  }
}

void div_op(Tensor *a, Tensor *b, Tensor *out) {
  int i = 0;
  int size = numel(a->shape, b->ndim);

  for (; i + 7 < size; i += 8) {
    __m256 x = _mm256_loadu_ps(a->data + i);
    __m256 y = _mm256_loadu_ps(b->data + i);
    __m256 z = _mm256_div_ps(x, y);
    _mm256_storeu_ps(out->data + i, z);
  }

  for (; i < size; ++i) {
    out->data[i] = a->data[i] / b->data[i];
  }

  out->requires_grad = a->requires_grad || b->requires_grad ? true : false;

  if (!out->requires_grad) {
    free_tensor(a);
    free_tensor(b);
  }
}

void matmul_op(Tensor *a, Tensor *b, Tensor *out, int N, int K, int P) {
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
    free(out->shape);
    free(out->strides);
    return;
  }

  int a_batch_stride = (a->ndim > 2) ? a->strides[a->ndim - 3] : N * K;
  int b_batch_stride = (b->ndim > 2) ? b->strides[b->ndim - 3] : K * P;
  int out_batch_stride = (out->ndim > 2) ? out->strides[out->ndim - 3] : N * P;

  const int simd_width = 8;
  const int k_simd = (K / simd_width) * simd_width;

  for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    int a_curr_stride = batch_idx * a_batch_stride;
    int b_curr_stride = batch_idx * b_batch_stride;
    int out_curr_stride = batch_idx * out_batch_stride;

    for (int i = 0; i < N; ++i) {
      int row = i * a->strides[a->ndim - 2];
      for (int j = 0; j < P; ++j) {
        int col = j * b->strides[b->ndim - 1];

        __m256 sum_vec = _mm256_setzero_ps();

        for (int k = 0; k < k_simd; k += simd_width) {
          __m256 a_vec = _mm256_loadu_ps(
              &a->data[a_curr_stride + row + k * a->strides[a->ndim - 1]]);
          __m256 b_vec = _mm256_loadu_ps(
              &b->data[b_curr_stride + col + k * b->strides[b->ndim - 2]]);
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

        for (int k = k_simd; k < K; ++k) {
          sum += a->data[a_curr_stride + row + k * a->strides[a->ndim - 1]] *
                 b->data[b_curr_stride + col + k * b->strides[b->ndim - 2]];
        }

        out->data[out_curr_stride + i * out->strides[out->ndim - 2] +
                  j * out->strides[out->ndim - 1]] = sum;
      }
    }
  }

  out->requires_grad = a->requires_grad || b->requires_grad;
  if (!out->requires_grad) {
    free_tensor(a);
    free_tensor(b);
  }
}
