#include "ops.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>

void add_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  if (n_prev == 2) {
    Tensor *a = prev[0];
    Tensor *b = prev[1];

    if (a->requires_grad) {
      int i = 0;
      int size = numel(a->shape, a->ndim);
      for (; i + 7 < size; i += 8) {
        __m256 b_data = _mm256_loadu_ps(b->data + i);
        __m256 dout = _mm256_loadu_ps(out->grad + i);
        __m256 da = _mm256_add_ps(b_data, dout);
        _mm256_storeu_ps(a->grad + i, da);
      }

      for (; i < size; ++i) {
        a->grad[i] += out->grad[i] * b->data[i];
      }
    }

    if (b->requires_grad) {
      int i = 0;
      int size = numel(a->shape, a->ndim);
      for (; i + 7 < size; i += 8) {
        __m256 a_data = _mm256_loadu_ps(a->data + i);
        __m256 dout = _mm256_loadu_ps(out->grad + i);
        __m256 db = _mm256_add_ps(a_data, dout);
        _mm256_storeu_ps(b->grad + i, db);
      }

      for (; i < size; ++i) {
        b->grad[i] += out->grad[i] * a->data[i];
      }
    }

  } else if (n_prev == 1 && extras != NULL) {
    Tensor *a = prev[0];
    float b = *((float *)extras);

    __m256 scalar = _mm256_set1_ps(b);

    if (a->requires_grad) {
      int i = 0;
      int size = numel(a->shape, a->ndim);
      for (; i + 7 < size; i += 8) {
        __m256 dout = _mm256_loadu_ps(out->grad + i);
        __m256 da = _mm256_add_ps(scalar, dout);
        _mm256_storeu_ps(a->grad + i, da);
      }

      for (; i < size; ++i) {
        a->grad[i] += out->grad[i] * b;
      }
    }
  }
}

void add_sub_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  if (n_prev == 2) {
    Tensor *a = prev[0];
    Tensor *b = prev[1];

    if (a->requires_grad) {
      int i = 0;
      int size = numel(a->shape, a->ndim);
      for (; i + 7 < size; i += 8) {
        __m256 b_data = _mm256_loadu_ps(b->data + i);
        __m256 dout = _mm256_loadu_ps(out->grad + i);
        __m256 da = _mm256_add_ps(b_data, dout);
        _mm256_storeu_ps(a->grad + i, da);
      }

      for (; i < size; ++i) {
        a->grad[i] += out->grad[i] * b->data[i];
      }
    }

    if (b->requires_grad) {
      int i = 0;
      int size = numel(a->shape, a->ndim);
      for (; i + 7 < size; i += 8) {
        __m256 a_data = _mm256_loadu_ps(a->data + i);
        __m256 dout = _mm256_loadu_ps(out->grad + i);
        __m256 db = _mm256_sub_ps(a_data, dout);
        _mm256_storeu_ps(b->grad + i, db);
      }

      for (; i < size; ++i) {
        b->grad[i] -= out->grad[i] * a->data[i];
      }
    }

  } else if (n_prev == 1 && extras != NULL) {
    Tensor *a = prev[0];
    float b = *((float *)extras);

    __m256 scalar = _mm256_set1_ps(b);

    if (a->requires_grad) {
      int i = 0;
      int size = numel(a->shape, a->ndim);
      for (; i + 7 < size; i += 8) {
        __m256 dout = _mm256_loadu_ps(out->grad + i);
        __m256 da = _mm256_add_ps(scalar, dout);
        _mm256_storeu_ps(a->grad + i, da);
      }

      for (; i < size; ++i) {
        a->grad[i] += out->grad[i] * b;
      }
    }
  }
}
