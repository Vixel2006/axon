#include "ops.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>

void relu_op(Tensor *in, Tensor *out) {
  int i = 0;
  int size = numel(in->shape, in->ndim);

  __m256 zeros = _mm256_setzero_ps();

  for (; i + 7 < size; i += 8) {
    __m256 vin = _mm256_loadu_ps(in->data + i);
    __m256 vout = _mm256_max_ps(vin, zeros);
    _mm256_storeu_ps(out->data + i, vout);
  }

  for (; i < size; ++i) {
    out->data[i] = in->data[i] > 0.0f ? in->data[i] : 0.0f;
  }

  out->requires_grad = in->requires_grad;

  if (!out->requires_grad) {
    free_tensor(in);
  }
}

void log_op(Tensor *in, Tensor *out) {
  int i = 0;
  int size = numel(in->shape, in->ndim);

  for (; i + 7 < size; i += 8) {
    __m256 x = _mm256_loadu_ps(in->data + i);
    __m256 y = Sleef_logf8_u10avx2(x);
    _mm256_storeu_ps(out->data + i, y);
  }

  for (; i < size; ++i) {
    out->data[i] = logf(in->data[i]);
  }

  out->requires_grad = in->requires_grad;

  if (!out->requires_grad) {
    free_tensor(in);
  }
}

void exp_op(Tensor *in, Tensor *out) {
  int i = 0;
  int size = numel(in->shape, in->ndim);

  for (; i + 7 < size; i += 8) {
    __m256 x = _mm256_loadu_ps(in->data + i);
    __m256 y = Sleef_expf8_u10avx2(x);
    _mm256_storeu_ps(out->data + i, y);
  }

  for (; i < size; ++i) {
    out->data[i] = expf(in->data[i]);
  }

  out->requires_grad = in->requires_grad;

  if (!out->requires_grad) {
    free_tensor(in);
  }
}

void softmax_op(Tensor *in, Tensor *out) {}

void neg_op(Tensor *in, Tensor *out) {
  int i = 0;
  int size = numel(in->shape, in->ndim);

  __m256 zeros = _mm256_setzero_ps();

  for (; i + 7 < size; i += 8) {
    __m256 x = _mm256_loadu_ps(in->data + i);
    __m256 y = _mm256_sub_ps(zeros, x);
    _mm256_storeu_ps(out->data + i, y);
  }

  for (; i < size; ++i) {
    out->data[i] = 0.0f - in->data[i];
  }

  out->requires_grad = in->requires_grad;

  if (!out->requires_grad) {
    free_tensor(in);
  }
}

void abs_op(Tensor *in, Tensor *out) {
  int i = 0;
  int size = numel(in->shape, in->ndim);
  __m256 mask = _mm256_castsi256_ps(
      _mm256_set1_epi32(0x7FFFFFFF)); // mask to remove sign bit

  for (; i + 7 < size; i += 8) {
    __m256 x = _mm256_loadu_ps(in->data + i);
    __m256 y = _mm256_and_ps(x, mask);
    _mm256_storeu_ps(out->data + i, y);
  }

  for (; i < size; ++i) {
    out->data[i] = in->data[i] >= 0 ? in->data[i] : 0.0f - in->data[i];
  }

  out->requires_grad = in->requires_grad;

  if (!out->requires_grad) {
    free_tensor(in);
  }
}
