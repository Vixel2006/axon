#include "ops.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>

void add_scalar_op(Tensor *a, float b, Tensor *out) {
  int i = 0;
  int size = numel(a->shape, a->ndim);

  __m256 scalar = _mm256_set1_ps(b);

  for (; i + 7 < size; i += 8) {
    __m256 x = _mm256_loadu_ps(a->data + i);
    __m256 z = _mm256_add_ps(x, scalar);
    _mm256_storeu_ps(out->data + i, z);
  }

  for (; i < size; ++i) {
    out->data[i] = a->data[i] + b;
  }

  out->requires_grad = a->requires_grad ? true : false;

  if (!out->requires_grad) {
    free_tensor(a);
  }
}

void sub_scalar_op(Tensor *a, float b, Tensor *out) {
  int i = 0;
  int size = numel(a->shape, a->ndim);

  __m256 scalar = _mm256_set1_ps(b);

  for (; i + 7 < size; i += 8) {
    __m256 x = _mm256_loadu_ps(a->data + i);
    __m256 z = _mm256_sub_ps(x, scalar);
    _mm256_storeu_ps(out->data + i, z);
  }

  for (; i < size; ++i) {
    out->data[i] = a->data[i] - b;
  }

  out->requires_grad = a->requires_grad ? true : false;

  if (!out->requires_grad) {
    free_tensor(a);
  }
}

void rsub_scalar_op(float a, Tensor *b, Tensor *out) {
  int i = 0;
  int size = numel(b->shape, b->ndim);

  __m256 scalar = _mm256_set1_ps(a);

  for (; i + 7 < size; i += 8) {
    __m256 x = _mm256_loadu_ps(b->data + i);
    __m256 z = _mm256_sub_ps(x, scalar);
    _mm256_storeu_ps(out->data + i, z);
  }

  for (; i < size; ++i) {
    out->data[i] = a - b->data[i];
  }

  out->requires_grad = b->requires_grad ? true : false;

  if (!out->requires_grad) {
    free_tensor(b);
  }
}

void mul_scalar_op(Tensor *a, float b, Tensor *out) {
  int i = 0;
  int size = numel(a->shape, a->ndim);

  __m256 scalar = _mm256_set1_ps(b);

  for (; i + 7 < size; i += 8) {
    __m256 x = _mm256_loadu_ps(a->data + i);
    __m256 z = _mm256_mul_ps(x, scalar);
    _mm256_storeu_ps(out->data + i, z);
  }

  for (; i < size; ++i) {
    out->data[i] = a->data[i] * b;
  }

  out->requires_grad = a->requires_grad ? true : false;

  if (!out->requires_grad) {
    free_tensor(a);
  }
}

void div_scalar_op(Tensor *a, float b, Tensor *out) {
  int i = 0;
  int size = numel(a->shape, a->ndim);

  __m256 scalar = _mm256_set1_ps(b);

  for (; i + 7 < size; i += 8) {
    __m256 x = _mm256_loadu_ps(a->data + i);
    __m256 z = _mm256_div_ps(x, scalar);
    _mm256_storeu_ps(out->data + i, z);
  }

  for (; i < size; ++i) {
    out->data[i] = a->data[i] / b;
  }

  out->requires_grad = a->requires_grad ? true : false;

  if (!out->requires_grad) {
    free_tensor(a);
  }
}

void rdiv_scalar_op(float a, Tensor *b, Tensor *out) {
  int i = 0;
  int size = numel(b->shape, b->ndim);

  __m256 scalar = _mm256_set1_ps(a);

  for (; i + 7 < size; i += 8) {
    __m256 x = _mm256_loadu_ps(b->data + i);
    __m256 z = _mm256_div_ps(scalar, x);
    _mm256_storeu_ps(out->data + i, z);
  }

  for (; i < size; ++i) {
    out->data[i] = a / b->data[i];
  }

  out->requires_grad = b->requires_grad ? true : false;

  if (!out->requires_grad) {
    free_tensor(b);
  }
}
