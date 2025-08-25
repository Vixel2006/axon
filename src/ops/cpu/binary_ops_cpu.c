#include "ops.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>

void add_op(Tensor *a, Tensor *b, Tensor *out) {
  int i = 0;
  int size = numel(a->shape, a->ndim);

  for (; i + 7 < size; i += 8) {
    __m256 x = _m256_loadu_ps(a->data + i);
    __m256 y = _m256_loadu_ps(b->data + i);
    __m256 z = _m256_add_ps(x, y);
    _m256_storeu_ps(out->data + i, z);
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
    __m256 x = _m256_loadu_ps(a->data + i);
    __m256 y = _m256_loadu_ps(b->data + i);
    __m256 z = _m256_sub_ps(x, y);
    _m256_storeu_ps(out->data + i, z);
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
    __m256 x = _m256_loadu_ps(a->data + i);
    __m256 y = _m256_loadu_ps(b->data + i);
    __m256 z = _m256_mul_ps(x, y);
    _m256_storeu_ps(out->data + i, z);
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
    __m256 x = _m256_loadu_ps(a->data + i);
    __m256 y = _m256_loadu_ps(b->data + i);
    __m256 z = _m256_div_ps(x, y);
    _m256_storeu_ps(out->data + i, z);
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

void matmul_op(Tensor *a, Tensor *b, Tensor *out) {
  int i = 0;
  int size = numel(a->shape, a->ndim);
}
