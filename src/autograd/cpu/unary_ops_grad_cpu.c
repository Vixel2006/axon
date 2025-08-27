#include "ops.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>

void relu_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *a = prev[0];

  int i = 0;
  int size = numel(a->shape, a->ndim);
  for (; i + 7 < size; i += 8) {
    __m256 va = _mm256_loadu_ps(a->data + i);
    __m256 mask = _mm256_cmp_ps(va, _mm256_setzero_ps(), _CMP_GT_OQ);
    __m256 dout = _mm256_loadu_ps(out->grad + i);
    __m256 dmasked = _mm256_and_ps(dout, mask);
    __m256 dprev = _mm256_loadu_ps(a->grad + i);
    dprev = _mm256_add_ps(dprev, dmasked);
    _mm256_storeu_ps(a->grad + i, dprev);
  }

  for (; i < size; ++i) {
    if (a->data[i] > 0) {
      a->grad[i] += out->grad[i];
    }
  }
}

void log_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *a = prev[0];

  int i = 0;
  int size = numel(a->shape, a->ndim);
  for (; i + 7 < size; i += 8) {
    __m256 va = _mm256_loadu_ps(a->data + i);
    __m256 dout = _mm256_loadu_ps(out->grad + i);
    __m256 da = _mm256_loadu_ps(a->grad + i);

    __m256 inv = _mm256_rcp_ps(va);
    __m256 contrib = _mm256_mul_ps(dout, inv);

    da = _mm256_add_ps(da, contrib);
    _mm256_storeu_ps(a->grad + i, da);
  }

  for (; i < size; ++i) {
    a->grad[i] += out->grad[i] / a->data[i];
  }
}

void exp_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *a = prev[0];

  int i = 0;
  int size = numel(a->shape, a->ndim);
  for (; i + 7 < size; i += 8) {
    __m256 vout = _mm256_loadu_ps(out->data + i);
    __m256 dout = _mm256_loadu_ps(out->grad + i);
    __m256 da = _mm256_loadu_ps(a->grad + i);

    __m256 contrib = _mm256_mul_ps(dout, vout);

    da = _mm256_add_ps(da, contrib);
    _mm256_storeu_ps(a->grad + i, da);
  }

  for (; i < size; ++i) {
    a->grad[i] += out->grad[i] * out->data[i];
  }
}

void softmax_grad_op(Tensor *in, Tensor *out) {}

void neg_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *a = prev[0];

  int size = numel(a->shape, a->ndim);
  int i = 0;

  __m256 neg_one = _mm256_set1_ps(-1.0f);

  for (; i + 7 < size; i += 8) {
    __m256 dout = _mm256_loadu_ps(out->grad + i);
    __m256 da = _mm256_loadu_ps(a->grad + i);

    __m256 contrib = _mm256_mul_ps(dout, neg_one);

    da = _mm256_add_ps(da, contrib);

    _mm256_storeu_ps(a->grad + i, da);
  }

  for (; i < size; ++i) {
    a->grad[i] += -out->grad[i];
  }
}

void abs_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *a = prev[0];
  int size = numel(a->shape, a->ndim);
  int i = 0;

  __m256 zero = _mm256_setzero_ps();
  __m256 one = _mm256_set1_ps(1.0f);
  __m256 neg1 = _mm256_set1_ps(-1.0f);

  for (; i + 7 < size; i += 8) {
    __m256 x = _mm256_loadu_ps(a->data + i);
    __m256 dout = _mm256_loadu_ps(out->grad + i);
    __m256 da = _mm256_loadu_ps(a->grad + i);

    __m256 mask_pos = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
    __m256 mask_neg = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);

    __m256 sign = _mm256_blendv_ps(zero, one, mask_pos);
    sign = _mm256_blendv_ps(sign, neg1, mask_neg);

    __m256 contrib = _mm256_mul_ps(dout, sign);

    da = _mm256_add_ps(da, contrib);

    _mm256_storeu_ps(a->grad + i, da);
  }

  for (; i < size; ++i) {
    float x = a->data[i];
    float s = (x > 0) ? 1.0f : (x < 0 ? -1.0f : 0.0f);
    a->grad[i] += out->grad[i] * s;
  }
}
