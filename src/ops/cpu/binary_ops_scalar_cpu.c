#include "ops/ops.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>

void add_scalar_op(Tensor *a, float b, Tensor *out) {
  int size = numel(a->shape, a->ndim);

  if (!is_contiguous(a) || !is_contiguous(out)) {
    for (int idx = 0; idx < size; ++idx) {
      int offset_a = 0;
      int offset_out = 0;
      int tmp = idx;

      for (int d = a->ndim - 1; d >= 0; --d) {
        int coord = tmp % a->shape[d];
        tmp /= a->shape[d];
        offset_a += coord * a->strides[d];
        offset_out += coord * out->strides[d];
      }

      out->data->ptr[offset_out] = a->data->ptr[offset_a] + b;
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(a->data->ptr + i);
      __m256 z = _mm256_add_ps(x, scalar);
      _mm256_storeu_ps(out->data->ptr + i, z);
    }

    for (; i < size; ++i) {
      out->data->ptr[i] = a->data->ptr[i] + b;
    }
  }

  out->requires_grad = a->requires_grad ? true : false;
}

void sub_scalar_op(Tensor *a, float b, Tensor *out) {
  int size = numel(a->shape, a->ndim);

  if (!is_contiguous(a) || !is_contiguous(out)) {
    for (int idx = 0; idx < size; ++idx) {
      int offset_a = 0;
      int offset_out = 0;
      int tmp = idx;

      for (int d = a->ndim - 1; d >= 0; --d) {
        int coord = tmp % a->shape[d];
        tmp /= a->shape[d];
        offset_a += coord * a->strides[d];
        offset_out += coord * out->strides[d];
      }

      out->data->ptr[offset_out] = a->data->ptr[offset_a] - b;
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(a->data->ptr + i);
      __m256 z = _mm256_sub_ps(x, scalar);
      _mm256_storeu_ps(out->data->ptr + i, z);
    }

    for (; i < size; ++i) {
      out->data->ptr[i] = a->data->ptr[i] - b;
    }
  }

  out->requires_grad = a->requires_grad ? true : false;
}

void rsub_scalar_op(float a, Tensor *b, Tensor *out) {
  int size = numel(b->shape, b->ndim);

  if (!is_contiguous(b) || !is_contiguous(out)) {
    for (int idx = 0; idx < size; ++idx) {
      int b_offset = 0;
      int out_offset = 0;
      int tmp = idx;

      for (int d = b->ndim - 1; d >= 0; --d) {
        int coord = tmp % b->shape[d];
        tmp /= b->shape[d];

        b_offset += coord * b->strides[d];
        out_offset += coord * out->strides[d];
      }

      out->data->ptr[out_offset] = a - b->data->ptr[b_offset];
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(a);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(b->data->ptr + i);
      __m256 z = _mm256_sub_ps(scalar, x);
      _mm256_storeu_ps(out->data->ptr + i, z);
    }

    for (; i < size; ++i) {
      out->data->ptr[i] = a - b->data->ptr[i];
    }
  }

  out->requires_grad = b->requires_grad ? true : false;
}

void mul_scalar_op(Tensor *a, float b, Tensor *out) {
  int size = numel(a->shape, a->ndim);

  if (!is_contiguous(a) || !is_contiguous(out)) {
    for (int idx = 0; idx < size; ++idx) {
      int offset_a = 0;
      int offset_out = 0;
      int tmp = idx;

      for (int d = a->ndim - 1; d >= 0; --d) {
        int coord = tmp % a->shape[d];
        tmp /= a->shape[d];
        offset_a += coord * a->strides[d];
        offset_out += coord * out->strides[d];
      }

      out->data->ptr[offset_out] = a->data->ptr[offset_a] * b;
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(a->data->ptr + i);
      __m256 z = _mm256_mul_ps(x, scalar);
      _mm256_storeu_ps(out->data->ptr + i, z);
    }

    for (; i < size; ++i) {
      out->data->ptr[i] = a->data->ptr[i] * b;
    }
  }

  out->requires_grad = a->requires_grad ? true : false;
}

void div_scalar_op(Tensor *a, float b, Tensor *out) {
  int size = numel(a->shape, a->ndim);

  if (!is_contiguous(a) || !is_contiguous(out)) {
    for (int idx = 0; idx < size; ++idx) {
      int offset_a = 0;
      int offset_out = 0;
      int tmp = idx;

      for (int d = a->ndim - 1; d >= 0; --d) {
        int coord = tmp % a->shape[d];
        tmp /= a->shape[d];
        offset_a += coord * a->strides[d];
        offset_out += coord * out->strides[d];
      }

      out->data->ptr[offset_out] = a->data->ptr[offset_a] / b;
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(a->data->ptr + i);
      __m256 z = _mm256_div_ps(x, scalar);
      _mm256_storeu_ps(out->data->ptr + i, z);
    }

    for (; i < size; ++i) {
      out->data->ptr[i] = a->data->ptr[i] / b;
    }
  }

  out->requires_grad = a->requires_grad ? true : false;
}

void rdiv_scalar_op(Tensor *a, float b, Tensor *out) {
  int size = numel(a->shape, a->ndim);

  if (!is_contiguous(a) || !is_contiguous(out)) {
    for (int idx = 0; idx < size; ++idx) {
      int offset_a = 0;
      int offset_out = 0;
      int tmp = idx;

      for (int d = a->ndim - 1; d >= 0; --d) {
        int coord = tmp % a->shape[d];
        tmp /= a->shape[d];
        offset_a += coord * a->strides[d];
        offset_out += coord * out->strides[d];
      }

      out->data->ptr[offset_out] = b / a->data->ptr[offset_a];
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(a->data->ptr + i);
      __m256 z = _mm256_div_ps(scalar, x);
      _mm256_storeu_ps(out->data->ptr + i, z);
    }

    for (; i < size; ++i) {
      out->data->ptr[i] = b / a->data->ptr[i];
    }
  }

  out->requires_grad = a->requires_grad ? true : false;
}

void pow_scalar_op(Tensor *a, float b, Tensor *out) {
  int size = numel(a->shape, a->ndim);

  if (!is_contiguous(a) || !is_contiguous(out)) {
    for (int idx = 0; idx < size; ++idx) {
      int offset_a = 0;
      int offset_out = 0;
      int tmp = idx;

      for (int d = a->ndim - 1; d >= 0; --d) {
        int coord = tmp % a->shape[d];
        tmp /= a->shape[d];
        offset_a += coord * a->strides[d];
        offset_out += coord * out->strides[d];
      }

      out->data->ptr[offset_out] = pow(a->data->ptr[offset_a], b);
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(a->data->ptr + i);
      __m256 y = Sleef_powf8_u10avx2(x, scalar);
      _mm256_storeu_ps(out->data->ptr + i, y);
    }

    for (; i < size; ++i) {
      out->data->ptr[i] = pow(a->data->ptr[i], b);
    }
  }
  out->requires_grad = a->requires_grad;
}
